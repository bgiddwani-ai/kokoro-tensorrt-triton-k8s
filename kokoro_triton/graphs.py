"""Restore standard ONNX operators and split the pinned batch-one graph.

Tensor names below are the public model's graph contract. A new upstream graph
requires reviewing that contract rather than silently guessing split points.
"""

import copy
from collections import Counter
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as T
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference


def make_portable(source, target=None):
    m = onnx.load(source)
    types = {
        v.name: v.type.tensor_type.elem_type
        for v in list(m.graph.value_info) + list(m.graph.input) + list(m.graph.output)
    }
    types.update({v.name: v.data_type for v in m.graph.initializer})
    serial = 0

    def rewrite(g):
        nonlocal serial
        out = []

        def op(kind, inputs, outputs, **attrs):
            nonlocal serial
            serial += 1
            out.append(
                h.make_node(
                    kind, inputs, outputs, name=f"portable_{serial}_{kind}", **attrs
                )
            )

        def c(value, ref):
            nonlocal serial
            serial += 1
            name = f"portable_constant_{serial}"
            dtype = np.float16 if types.get(ref) == T.FLOAT16 else np.float32
            g.initializer.append(nh.from_array(np.array(value, dtype=dtype), name))
            return name

        for n in g.node:
            for a in n.attribute:
                if a.type == onnx.AttributeProto.GRAPH:
                    rewrite(a.g)
            a = {a.name: h.get_attribute_value(a) for a in n.attribute}
            x = list(n.input)
            y = list(n.output)
            tag = f"portable_{serial}_"
            if n.op_type in ("MemcpyToHost", "MemcpyFromHost"):
                op("Identity", x, y)
            elif n.op_type == "FusedMatMul":
                for i, k in enumerate(("A", "B")):
                    assert not a.get("transBatch" + k, 0)
                    if a.get("trans" + k, 0):
                        rank = 3
                        vi = next(
                            (
                                v
                                for v in list(g.input)
                                + list(g.value_info)
                                + list(m.graph.value_info)
                                if v.name == x[i]
                            ),
                            None,
                        )
                        if vi:
                            rank = len(vi.type.tensor_type.shape.dim)
                        perm = list(range(rank))
                        perm[-1], perm[-2] = perm[-2], perm[-1]
                        z = tag + f"transpose{i}"
                        op("Transpose", [x[i]], [z], perm=perm)
                        x[i] = z
                alpha = a.get("alpha", 1.0)
                if alpha == 1:
                    op("MatMul", x, y)
                else:
                    z = tag + "matmul"
                    op("MatMul", x, [z])
                    op("Mul", [z, c(alpha, n.input[0])], y)
            elif n.op_type == "FastGelu":
                z = x[0]
                if len(x) > 1 and x[1]:
                    z = tag + "bias"
                    op("Add", x[:2], [z])
                z2 = tag + "square"
                z3 = tag + "cube"
                p = tag + "p"
                q = tag + "q"
                r = tag + "r"
                s = tag + "s"
                t = tag + "t"
                u = tag + "u"
                op("Mul", [z, z], [z2])
                op("Mul", [z2, z], [z3])
                op("Mul", [z3, c(0.044715, n.input[0])], [p])
                op("Add", [p, z], [q])
                op("Mul", [q, c(0.7978845608028654, n.input[0])], [r])
                op("Tanh", [r], [s])
                op("Add", [s, c(1.0, n.input[0])], [t])
                op("Mul", [z, c(0.5, n.input[0])], [u])
                op("Mul", [t, u], y)
            elif n.op_type == "SkipLayerNormalization":
                z = tag + "sum"
                op("Add", x[:2], [z])
                if len(x) > 4 and x[4]:
                    z2 = tag + "bias"
                    op("Add", [z, x[4]], [z2])
                    z = z2
                assert len(y) == 1
                op(
                    "LayerNormalization",
                    [z, x[2], x[3]],
                    y,
                    axis=-1,
                    epsilon=a.get("epsilon", 1e-5),
                )
            else:
                out.append(n)
        del g.node[:]
        g.node.extend(out)

    rewrite(m.graph)
    print("saved portable", len(m.graph.node), Counter(n.op_type for n in m.graph.node))
    onnx.checker.check_model(m)

    # Both axis-free squeezes remove only the fixed batch/channel dimensions.
    # Batch size is one in this upstream graph.
    for n in m.graph.node:
        if n.op_type == "Squeeze" and len(n.input) == 1:
            name = n.name + "_flatten_shape"
            m.graph.initializer.append(nh.from_array(np.array([-1], np.int64), name))
            n.op_type = "Reshape"
            n.input.append(name)
    onnx.checker.check_model(m)
    if target is not None:
        onnx.save(m, str(target))
    return m


def split_model(source):
    m = source if isinstance(source, onnx.ModelProto) else onnx.load(source)

    # Promote upstream mixed-precision weights/activations to FP32.
    def promote(g):
        for n in g.node:
            if n.op_type == "Cast":
                for a in n.attribute:
                    if a.name == "to" and a.i == T.FLOAT16:
                        a.i = T.FLOAT
            for a in n.attribute:
                if a.type == onnx.AttributeProto.GRAPH:
                    promote(a.g)
                elif (
                    a.type == onnx.AttributeProto.TENSOR and a.t.data_type == T.FLOAT16
                ):
                    a.t.CopyFrom(
                        nh.from_array(nh.to_array(a.t).astype(np.float32), a.t.name)
                    )
        for t in g.initializer:
            if t.data_type == T.FLOAT16:
                t.CopyFrom(nh.from_array(nh.to_array(t).astype(np.float32), t.name))
        for v in list(g.input) + list(g.output) + list(g.value_info):
            if v.type.tensor_type.elem_type == T.FLOAT16:
                v.type.tensor_type.elem_type = T.FLOAT

    promote(m.graph)
    producers = {o: i for i, n in enumerate(m.graph.node) for o in n.output}
    weights = {t.name: t for t in m.graph.initializer}

    def dependencies(n):
        vals = set(x for x in n.input if x)
        for a in n.attribute:
            if a.type == onnx.AttributeProto.GRAPH:
                defined = (
                    {v.name for v in a.g.input}
                    | {v.name for v in a.g.initializer}
                    | {o for sn in a.g.node for o in sn.output}
                )
                vals |= {
                    x for sn in a.g.node for x in dependencies(sn) if x not in defined
                }
        return vals

    def extract(name, inputs, outputs):
        seen = set()
        needed = set()
        stop = {x[0] for x in inputs}

        def visit(t):
            if t in stop:
                return
            if t in weights:
                needed.add(t)
                return
            if t not in producers:
                raise ValueError(("missing", t))
            ix = producers[t]
            if ix in seen:
                return
            seen.add(ix)
            for x in dependencies(m.graph.node[ix]):
                visit(x)

        for x in outputs:
            visit(x[0])
        nodes = [copy.deepcopy(n) for i, n in enumerate(m.graph.node) if i in seen]
        graph = h.make_graph(
            nodes,
            name,
            [h.make_tensor_value_info(*x) for x in inputs],
            [h.make_tensor_value_info(*x) for x in outputs],
            [weights[t] for t in sorted(needed)],
        )
        model = h.make_model(
            graph,
            opset_imports=m.opset_import,
            ir_version=m.ir_version,
            producer_name="kokoro_portable_split",
        )
        onnx.checker.check_model(model)
        print(name, len(nodes), len(needed), flush=True)
        return model

    p_en = "/text_encoder/Where_13"
    p_te = "/text_encoder_1/Where_7_output_0"
    dur = "onnx::CumSum_3090"
    en = "/MatMul_output_0"
    te = "/MatMul_1_output_0"
    predictor = extract(
        "predictor",
        [
            ("tokens", T.INT64, [1, "S"]),
            ("style", T.FLOAT, [1, 256]),
            ("speed", T.FLOAT, [1]),
        ],
        [
            (dur, T.INT64, ["S"]),
            (p_en, T.FLOAT, [1, "S", 640]),
            (p_te, T.FLOAT, [1, 512, "S"]),
        ],
    )
    decoder = extract(
        "decoder",
        [
            (en, T.FLOAT, [1, 640, "F"]),
            (te, T.FLOAT, [1, 512, "F"]),
            ("style", T.FLOAT, [1, 256]),
        ],
        [("audio", T.FLOAT, ["N"])],
    )
    return {"predictor": predictor, "decoder": decoder}


def infer_partition_shapes(source, target=None):
    # This specialization is valid for the pinned, fixed-batch-one upstream graph.
    m = source if isinstance(source, onnx.ModelProto) else onnx.load(source)
    name = m.graph.name
    nodes = []
    for n in m.graph.node:
        if n.op_type == "If":
            branch = next(a.g for a in n.attribute if a.name == "then_branch")
            assert len(branch.node) == 1 and branch.node[0].op_type == "Squeeze"
            m.graph.initializer.extend(copy.deepcopy(list(branch.initializer)))
            q = copy.deepcopy(branch.node[0])
            q.output[0] = n.output[0]
            nodes.append(q)
        else:
            nodes.append(n)
    del m.graph.node[:]
    m.graph.node.extend(nodes)
    m = onnx.shape_inference.infer_shapes(m, strict_mode=False, data_prop=True)
    try:
        m = SymbolicShapeInference.infer_shapes(
            copy.deepcopy(m), auto_merge=True, guess_output_rank=True, verbose=0
        )
    except Exception as e:
        print("symbolic inference:", name, str(e), flush=True)
    onnx.checker.check_model(m)
    if target is not None:
        onnx.save(m, str(target))
    print("SHAPES_SAVED", name, len(m.graph.value_info), flush=True)
    return m
