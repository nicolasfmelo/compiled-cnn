#!/usr/bin/env python3
"""Compile the static raw ONNX graph into fixed-shape C++ calls and an arena plan.

No inference library is used by the generated executable. Unknown operators,
attributes, dynamic dimensions and unsupported parameter combinations are errors.
"""
import argparse
from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np
import onnx
from onnx import numpy_helper, helper

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Value:
    shape: tuple
    constant: object = None
    alias: object = None
    offset: int = 0


def prod(s):
    return math.prod(s)


def compile_graph(source, destination, stage):
    model = onnx.load(source)
    if any(n.domain not in ('', 'ai.onnx') for n in model.graph.node):
        raise ValueError('custom ONNX domains are unsupported')
    if any(x.type.tensor_type.elem_type != onnx.TensorProto.FLOAT for x in model.graph.input):
        raise ValueError('only FP32 graph inputs are supported')
    onnx.checker.check_model(model, full_check=True)
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError('exactly one input/output required')
    values = {}
    for x in model.graph.input:
        shape = tuple(d.dim_value for d in x.type.tensor_type.shape.dim)
        if shape != (1, 3, 640, 640):
            raise ValueError(f'unsupported input {shape}')
        values[x.name] = Value(shape)
    for x in model.graph.initializer:
        a = numpy_helper.to_array(x)
        values[x.name] = Value(a.shape, a)
    ops = []
    # All accepted attributes are explicitly checked below; even ignored defaults
    # must have their specified ONNX value.
    allowed = dict(Constant={'value'}, Conv={'dilations', 'group', 'kernel_shape', 'pads', 'strides', 'auto_pad'},
                   Sigmoid=set(), Mul=set(), Add=set(), Sub=set(), Div=set(), Shape={'start', 'end'},
                   Gather={'axis'}, Slice=set(), Concat={'axis'}, Split={'axis', 'split', 'num_outputs'},
                   Reshape={'allowzero'}, Transpose={'perm'}, MaxPool={'ceil_mode', 'dilations', 'kernel_shape', 'pads', 'strides', 'auto_pad', 'storage_order'},
                   Resize={'coordinate_transformation_mode', 'cubic_coeff_a', 'mode', 'nearest_mode'},
                   MatMul=set(), Softmax={'axis'})
    def const(name):
        v = values[name]
        if v.constant is None:
            raise ValueError(f'expected compile-time constant: {name}')
        return v.constant
    for node in model.graph.node:
        kind = node.op_type
        a = {x.name: helper.get_attribute_value(x) for x in node.attribute}
        if kind not in allowed or set(a) - allowed[kind]:
            raise ValueError(f'{node.name}: unsupported op/attributes {kind} {a}')
        ins = [x for x in node.input if x]
        shapes = [values[x].shape for x in ins]
        result = None
        alias = None
        params = {}
        if kind == 'Constant':
            result = numpy_helper.to_array(a['value'])
            out_shapes = [result.shape]
        elif kind == 'Shape':
            result = np.array(shapes[0][a.get('start', 0):a.get('end', len(shapes[0]))], dtype=np.int64)
            out_shapes = [result.shape]
        elif kind == 'Gather':
            result = np.take(const(ins[0]), const(ins[1]), axis=a.get('axis', 0))
            out_shapes = [result.shape]
        elif kind in ('Add', 'Sub', 'Mul', 'Div'):
            shape = np.broadcast_shapes(*shapes)
            out_shapes = [shape]
            if all(values[x].constant is not None for x in ins):
                l, r = (const(x) for x in ins)
                if kind == 'Div' and np.issubdtype(l.dtype, np.integer):
                    result = np.trunc(l / r).astype(l.dtype)
                else:
                    result = {'Add': np.add, 'Sub': np.subtract, 'Mul': np.multiply, 'Div': np.divide}[kind](l, r)
        elif kind == 'Conv':
            n, c, h, w = shapes[0]
            oc, ic, kh, kw = shapes[1]
            group = a.get('group', 1)
            pads = a.get('pads', [0]*4)
            stride = a.get('strides', [1, 1])
            if pads[0] != pads[2] or pads[1] != pads[3]:
                raise ValueError(f'{node.name}: asymmetric Conv padding unsupported: {pads}')
            if (n != 1 or c != ic*group or oc % group or a.get('dilations', [1, 1]) != [1, 1]
                    or a.get('auto_pad', b'NOTSET') != b'NOTSET' or a.get('kernel_shape', [kh, kw]) != [kh, kw]):
                raise ValueError(f'{node.name}: unsupported Conv {a} {shapes}')
            oh, ow = (h+pads[0]+pads[2]-kh)//stride[0]+1, (w+pads[1]+pads[3]-kw)//stride[1]+1
            out_shapes = [(n, oc, oh, ow)]
            params = dict(kh=kh, kw=kw, sy=stride[0], sx=stride[1], py=pads[0], px=pads[1], group=group, silu=0)
            if len(ins) == 2:
                name = node.name + '/zero_bias'
                values[name] = Value((oc,), np.zeros(oc, np.float32))
                ins.append(name)
            params['macs'] = oc*oh*ow*ic*kh*kw
            k = ic*kh*kw
            tile = 32 if k > 2048 else 64
            spatial_jobs = (oh*ow+tile-1)//tile
            splits = max(1, (24+spatial_jobs-1)//spatial_jobs)
            chunk = ((oc+splits-1)//splits+5)//6*6
            # Full B tile plus the learned-weight slice below 1 MiB, leaving
            # 256 KiB of L2 for output/metadata and other activity.
            cache_rows = max(6, ((1024*1024//(4*k)-tile)//6)*6)
            params.update(spatial_tile=tile, output_chunk=min(chunk, cache_rows))
        elif kind == 'Sigmoid':
            out_shapes = [shapes[0]]
        elif kind == 'MaxPool':
            if (a.get('ceil_mode', 0) != 0 or a.get('dilations', [1, 1]) != [1, 1]
                    or a.get('auto_pad', b'NOTSET') != b'NOTSET' or a.get('storage_order', 0) != 0
                    or a['kernel_shape'] != [5, 5] or a['pads'] != [2]*4 or a['strides'] != [1, 1] or len(node.output) != 1):
                raise ValueError(f'{node.name}: unsupported MaxPool {a}')
            out_shapes = [shapes[0]]
        elif kind == 'Resize':
            if (a.get('mode', b'nearest') != b'nearest' or a.get('coordinate_transformation_mode', b'half_pixel') != b'asymmetric'
                    or a.get('nearest_mode', b'round_prefer_floor') != b'floor' or len(node.input) != 3
                    or not np.array_equal(const(node.input[2]), [1, 1, 2, 2])):
                raise ValueError(f'{node.name}: unsupported Resize {a}')
            n, c, h, w = shapes[0]
            out_shapes = [(n, c, h*2, w*2)]
            ins = ins[:1]
        elif kind == 'Reshape':
            if a.get('allowzero', 0) != 0:
                raise ValueError('Reshape allowzero unsupported')
            target = [int(x) for x in const(ins[1])]
            target = [shapes[0][i] if x == 0 else x for i, x in enumerate(target)]
            if target.count(-1) > 1:
                raise ValueError('multiple inferred Reshape dimensions')
            if -1 in target:
                target[target.index(-1)] = prod(shapes[0]) // -prod(target)
            assert prod(target) == prod(shapes[0])
            out_shapes = [tuple(target)]
            alias = [(ins[0], 0)]
            ins = ins[:1]
        elif kind == 'Transpose':
            perm = a.get('perm', list(reversed(range(len(shapes[0])))))
            assert sorted(perm) == list(range(len(shapes[0])))
            out_shapes = [tuple(shapes[0][i] for i in perm)]
            params = dict(perm=perm)
        elif kind in ('Slice', 'Split'):
            shape = shapes[0]
            if kind == 'Slice':
                starts, ends = const(ins[1]), const(ins[2])
                axes = const(ins[3]) if len(ins) > 3 else np.arange(len(starts))
                steps = const(ins[4]) if len(ins) > 4 else np.ones(len(starts), dtype=int)
                if len(axes) != 1 or steps[0] != 1:
                    raise ValueError('only single-axis unit-step Slice supported')
                axis = int(axes[0]) % len(shape)
                start, stop, _ = slice(int(starts[0]), int(ends[0])).indices(shape[axis])
                segments = [(start, max(0, stop-start))]
            else:
                axis = a.get('axis', 0) % len(shape)
                sizes = list(map(int, const(ins[1]))) if len(ins) > 1 else a.get('split')
                if sizes is None or sum(sizes) != shape[axis]:
                    raise ValueError('explicit Split sizes required')
                segments, start = [], 0
                for size in sizes:
                    segments.append((start, size)); start += size
            out_shapes = [shape[:axis]+(size,)+shape[axis+1:] for start, size in segments]
            if prod(shape[:axis]) == 1:
                alias = [(ins[0], start*prod(shape[axis+1:])) for start, size in segments]
            params = dict(axis=axis, segments=segments)
            ins = ins[:1]
        elif kind == 'Concat':
            axis = a['axis'] % len(shapes[0])
            assert all(s[:axis]+s[axis+1:] == shapes[0][:axis]+shapes[0][axis+1:] for s in shapes)
            out_shapes = [shapes[0][:axis]+(sum(s[axis] for s in shapes),)+shapes[0][axis+1:]]
            params = dict(axis=axis)
        elif kind == 'MatMul':
            assert len(shapes[0]) == len(shapes[1]) == 4
            assert shapes[0][:-2] == shapes[1][:-2] and shapes[0][-1] == shapes[1][-2]
            out_shapes = [shapes[0][:-1]+(shapes[1][-1],)]
            params['macs'] = prod(shapes[0])*shapes[1][-1]
        elif kind == 'Softmax':
            out_shapes = [shapes[0]]
            params = dict(axis=a.get('axis', -1) % len(shapes[0]))
        else:
            raise ValueError(f'unhandled {kind}')
        if alias:
            for base, _ in alias:
                while values[base].alias is not None:
                    base = values[base].alias
                if base == model.graph.input[0].name:
                    raise ValueError(f'{node.name}: input-rooted aliases unsupported')
        for i, (name, shape) in enumerate(zip(node.output, out_shapes)):
            if any(x <= 0 for x in shape):
                raise ValueError(f'{node.name}: invalid shape {shape}')
            values[name] = Value(tuple(shape), result)
            if alias:
                values[name].alias, values[name].offset = alias[i]
        if result is None and not alias:
            ops.append(dict(kind=kind, name=node.name, inputs=ins, outputs=list(node.output), **params))
    # Supply independently inferred static shapes, and replace the folded shape
    # subgraphs by initializers before asking ONNX to verify every dimension.
    folded = {k: v for k, v in values.items() if v.constant is not None}
    checked = onnx.ModelProto(); checked.CopyFrom(model)
    del checked.graph.node[:]
    for node in model.graph.node:
        if not all(x in folded for x in node.output):
            checked.graph.node.append(node)
    del checked.graph.initializer[:]
    checked.graph.initializer.extend(numpy_helper.from_array(v.constant, k) for k, v in folded.items())
    del checked.graph.value_info[:]
    inferred = onnx.shape_inference.infer_shapes(checked, strict_mode=True, data_prop=True)
    for x in list(inferred.graph.value_info)+list(inferred.graph.output)+list(inferred.graph.input):
        actual = values[x.name].shape
        expected = tuple(d.dim_value for d in x.type.tensor_type.shape.dim)
        if expected != actual:
            raise ValueError(f'shape disagreement {x.name}: ours={actual}, ONNX={expected}')
    def fused_root(name):
        while values[name].alias is not None:
            name = values[name].alias
        return name

    def consumers():
        uses = {k: [] for k in values}
        for op in ops:
            for x in op['inputs']: uses[fused_root(x)].append(op)
        # A graph output is observable even if no generated op consumes it.
        uses[fused_root(model.graph.output[0].name)].append({'kind': 'GraphOutput'})
        return uses

    # Recognize Conv -> Sigmoid -> Mul once; retain a separate SiLU in M1/M2.
    uses = consumers()
    removed = set()
    for op in ops:
        if op['kind'] != 'Sigmoid': continue
        x, sig = op['inputs'][0], op['outputs'][0]
        if len(uses[fused_root(sig)]) != 1: continue
        mul = uses[fused_root(sig)][0]
        if mul['kind'] != 'Mul' or set(mul['inputs']) != {x, sig}: continue
        op.update(kind='Silu', inputs=[x], outputs=mul['outputs'])
        removed.add(id(mul))
    ops = [op for op in ops if id(op) not in removed]
    # Generated references keep their original aliases, so rewrite by creating a
    # view from the fused producer output to the final activation/add output.
    if stage >= 3:
        uses = consumers()
        removed = set()
        for op in ops:
            if op['kind'] != 'Conv': continue
            out = op['outputs'][0]
            if (len(uses[out]) == 1 and uses[out][0]['kind'] == 'Silu'
                    and uses[out][0]['inputs'] == [out]):
                act = uses[out][0]
                values[act['outputs'][0]].alias = out
                op['silu'] = 1
                removed.add(id(act))
        ops = [op for op in ops if id(op) not in removed]
        uses = consumers()
        producer = {fused_root(x): i for i, op in enumerate(ops) for x in op['outputs']}
        removed = set()
        for i, op in enumerate(ops):
            if op['kind'] != 'Conv': continue
            out = op['outputs'][0]
            if len(uses[out]) != 1 or uses[out][0]['kind'] != 'Add': continue
            add = uses[out][0]
            if values[add['outputs'][0]].shape != values[out].shape: continue
            residual = next((x for x in add['inputs'] if fused_root(x) != out), None)
            if residual is None or values[residual].shape != values[out].shape: continue
            # A residual must already exist at the Conv, not merely at the Add.
            # Constants and graph inputs have no producing op and are ready.
            if producer.get(fused_root(residual), -1) >= i: continue
            op['residual'] = residual
            # Include the extra read in liveness and descriptor generation.
            op['inputs'].append(residual)
            values[add['outputs'][0]].alias = out
            removed.add(id(add))
        ops = [op for op in ops if id(op) not in removed]
    # Resolve alias roots before allocation. Contiguous Concat can become a
    # union of disjoint producer slices if none of those roots is shared/repeated.
    def root(name):
        off = 0
        seen = set()
        while values[name].alias is not None:
            assert name not in seen, 'alias cycle'
            seen.add(name)
            off += values[name].offset
            name = values[name].alias
        return name, off
    views = []
    if stage >= 3:
        for op in ops:
            if op['kind'] != 'Concat': continue
            axis = op['axis']
            if prod(values[op['outputs'][0]].shape[:axis]) != 1: continue
            groups = {}
            pos = 0
            ok = True
            for x in op['inputs']:
                r, off = root(x)
                if values[r].constant is not None or r == model.graph.input[0].name:
                    ok = False; break
                base = pos-off
                if base < 0 or (r in groups and groups[r] != base): ok = False; break
                groups[r] = base
                pos += prod(values[x].shape)
            intervals = sorted((base, base+prod(values[r].shape)) for r, base in groups.items())
            if any(b > c for (_, b), (c, _) in zip(intervals, intervals[1:])) or any(b > pos for _, b in intervals): ok = False
            if not ok: continue
            dst = op['outputs'][0]
            for r, base in groups.items():
                values[r].alias, values[r].offset = dst, base
            op['view'] = True
            views.append(op['name'])
        ops = [op for op in ops if not op.get('view')]
    used = {model.graph.input[0].name, model.graph.output[0].name}
    for op in ops: used.update(op['inputs']+op['outputs'])
    names = sorted(used)
    ids = {k: i for i, k in enumerate(names)}
    lifetimes = {}
    for i, op in enumerate(ops):
        for x in op['inputs']+op['outputs']:
            r, _ = root(x)
            if values[r].constant is not None or r == model.graph.input[0].name: continue
            span = lifetimes.setdefault(r, [i, i])
            span[1] = i
    outroot, _ = root(model.graph.output[0].name)
    lifetimes[outroot][1] = len(ops)
    offsets = {}
    free, active = [], []
    arena_size = 0
    for r, (start, end) in sorted(lifetimes.items(), key=lambda item: (item[1][0], item[0])):
        size = (prod(values[r].shape)+15)//16*16
        if stage >= 3:
            expired = [x for x in active if x[0] < start]
            active = [x for x in active if x[0] >= start]
            free.extend((o, n) for _, o, n in expired)
            merged = []
            for o, n in sorted(free):
                if merged and sum(merged[-1]) == o:
                    merged[-1] = (merged[-1][0], merged[-1][1]+n)
                else: merged.append((o, n))
            free = merged
        fit = next((i for i, (_, n) in enumerate(free) if n >= size), None)
        if fit is None:
            offset = arena_size; arena_size += size
        else:
            offset, n = free.pop(fit)
            if n > size: free.append((offset+size, n-size))
        offsets[r] = offset
        active.append((end, offset, size))
    destination.mkdir(parents=True, exist_ok=True)
    blob = bytearray()
    weights = {}
    def pack(name, array):
        if name in weights: return weights[name]
        blob.extend(bytes((-len(blob)) % 64))
        weights[name] = len(blob)//4
        blob.extend(np.asarray(array, dtype='<f4').tobytes())
        return weights[name]
    for name in names:
        r, _ = root(name)
        if values[r].constant is not None: pack(r, values[r].constant)
    # M5 packs six output rows, K major, with zero-padded channel tails.
    # Must match runtime CONV_MR; no inference-time weight packing.
    mr = 6
    for op in ops:
        if op['kind'] == 'Conv':
            w = const(op['inputs'][1]).reshape(values[op['inputs'][1]].shape[0], -1)
            oc, k = w.shape
            packed = np.zeros(((oc+mr-1)//mr, k, mr), np.float32)
            for c in range(oc): packed[c//mr, :, c%mr] = w[c]
            op['packed'] = pack(op['name']+'/packed', packed)
    (destination/'weights.bin').write_bytes(blob)
    (destination/'weights.S').write_text('.section .rodata\n.balign 64\n.global engine_weights\n.hidden engine_weights\nengine_weights:\n.incbin "weights.bin"\n.section .note.GNU-stack,"",@progbits\n')
    header = '''#pragma once
#include "runtime.h"
namespace engine {
class Model {
    Arena arena_;
    std::vector<Tensor> t_;
public:
    Context ctx;
    explicit Model(bool optimized, int threads = 1, const std::vector<int>& cpus = {0});
    const float* forward(const float* input);
    static double flops();
};
}
'''
    (destination/'model.h').write_text(header)
    lines = ['#include "model.h"', 'extern "C" const float engine_weights[];', 'namespace engine {',
             f'Model::Model(bool optimized, int threads, const std::vector<int>& cpus) : arena_({arena_size}), t_({len(names)}), ctx(optimized, threads, cpus) {{']
    for name in names:
        v = values[name]; r, off = root(name)
        if r == model.graph.input[0].name: pointer = 'nullptr'
        elif values[r].constant is not None: pointer = f'const_cast<float*>(engine_weights + {weights[r]+off})'
        else: pointer = f'arena_.data() + {offsets[r]+off}'
        lines.append(f' t_[{ids[name]}] = Tensor{{{pointer}, {{{", ".join(map(str, v.shape))}}}}};')
    lines += ['}', 'const float* Model::forward(const float* input) {', ' ThreadPool::Session session(ctx.pool);', f' t_[{ids[model.graph.input[0].name]}].data = const_cast<float*>(input);']
    plan = [f'stage: M{stage}', f'arena bytes: {arena_size*4}', f'weights bytes: {len(blob)}', 'layout: contiguous NCHW (ONNX row-major for attention/head)', 'Resize: nearest/asymmetric/floor x2', f'zero-copy concats: {len(views)}']
    macs = sum(op.get('macs', 0) for op in ops)
    plan += [f'MACs: {macs}', f'FLOPs (2 x MACs): {macs*2}', '']
    plan += [f'concat view: {name}' for name in views]
    for name in names:
        r, off = root(name)
        if values[r].constant is None and r in offsets:
            plan.append(f'buffer {name}: shape={values[name].shape} root={r} offset_floats={offsets[r]+off} root_live_ops={lifetimes[r]}')
    plan.append('')
    def t(x): return f't_[{ids[x]}]'
    for i, op in enumerate(ops):
        kind, ins, outs = op['kind'], op['inputs'], op['outputs']
        x, y = t(ins[0]), t(outs[0])
        category = 'other'
        if kind == 'Conv':
            args = ', '.join(str(op[k]) for k in ('kh', 'kw', 'sy', 'sx', 'py', 'px', 'group', 'silu', 'spatial_tile', 'output_chunk'))
            residual = '&'+t(op['residual']) if 'residual' in op else 'nullptr'
            call = f'conv(ctx, {x}, {t(ins[1])}, {t(ins[2])}, {y}, {{{args}}}, engine_weights + {op["packed"]}, {residual})'
            category = 'dwconv' if op['group'] > 1 else 'conv1x1' if op['kh'] == 1 else 'conv3x3s2' if op['sy'] == 2 else 'conv3x3'
        elif kind in ('Silu', 'Sigmoid'):
            call = f'unary(ctx, {x}, {y}, {1 if kind == "Silu" else 0})'
        elif kind in ('Add', 'Sub', 'Mul', 'Div'):
            call = f'binary(ctx, {x}, {t(ins[1])}, {y}, {("Add", "Sub", "Mul", "Div").index(kind)})'
        elif kind == 'MaxPool': call = f'maxpool(ctx, {x}, {y})'; category = 'maxpool'
        elif kind == 'Resize': call = f'resize(ctx, {x}, {y})'; category = 'resize'
        elif kind == 'Concat':
            call = f'concat(ctx, {{{", ".join("&"+t(z) for z in ins)}}}, {y}, {op["axis"]})'; category = 'concat/copy'
        elif kind in ('Split', 'Slice'):
            pieces = ', '.join('{'+f'{start}, {size}'+'}' for start, size in op['segments'])
            call = f'split(ctx, {x}, {{{", ".join("&"+t(z) for z in outs)}}}, {op["axis"]}, {{{pieces}}})'; category = 'concat/copy'
        elif kind == 'Transpose':
            call = f'transpose(ctx, {x}, {y}, {{{", ".join(map(str, op["perm"]))}}})'; category = 'concat/copy'
        elif kind == 'MatMul': call = f'matmul(ctx, {x}, {t(ins[1])}, {y})'; category = 'attention'
        elif kind == 'Softmax': call = f'softmax(ctx, {x}, {y}, {op["axis"]})'; category = 'attention'
        else: raise ValueError(kind)
        if '/model.23/' in op['name'] and kind != 'Conv': category = 'head'
        lines.append(f' ctx.run("{category}", [&] {{ {call}; }}); // {op["name"]}')
        desc = ', '.join(f'{z}:{values[z].shape}' for z in ins)
        buffers = ', '.join(f'{z}=arena[{offsets.get(root(z)[0], -1)+root(z)[1]}]' for z in outs)
        if kind == 'Conv': plan.append(f'grid {op["name"]}: spatial_tile={op["spatial_tile"]} output_chunk={op["output_chunk"]}')
        plan.append(f'{i:03d} {kind} {op["name"]} ({desc}) -> {[values[z].shape for z in outs]} MACs={op.get("macs", 0)} {buffers}')
    lines += [f' return t_[{ids[model.graph.output[0].name]}].data;', '}', f'double Model::flops() {{ return {macs*2}.0; }}', '}']
    (destination/'model.cpp').write_text('\n'.join(lines)+'\n')
    (destination/'plan.txt').write_text('\n'.join(plan)+'\n')
    print(f'Generated M{stage}: {len(ops)} calls, {macs*2} FLOPs, arena {arena_size*4} bytes, {len(views)} concat views; all shapes verified against ONNX')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=ROOT/'artifacts/engine/yolov10s_raw.onnx')
    p.add_argument('--output', type=Path, default=ROOT/'artifacts/engine/generated')
    p.add_argument('--stage', type=int, choices=(1, 2, 3, 4, 5), default=5)
    args = p.parse_args()
    try:
        compile_graph(args.input, args.output, args.stage)
    except (ValueError, AssertionError, KeyError, onnx.checker.ValidationError, onnx.shape_inference.InferenceError) as error:
        p.exit(1, f'Graph compilation failed: {error}\n')


if __name__ == '__main__':
    main()
