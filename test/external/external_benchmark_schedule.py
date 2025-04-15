from typing import List

from extra.models.resnet import ResNet50
from tinygrad import Device, Tensor, nn
from tinygrad.codegen.devectorizer import full_graph_rewrite
from tinygrad.codegen.heuristic import hand_coded_optimizations
from tinygrad.codegen.kernel import Kernel
from tinygrad.codegen.linearize import linearize_uop
from tinygrad.codegen.lowerer import rewrite_shapetracker_with_index
from tinygrad.engine.search import beam_search, bufs_from_lin
from tinygrad.helpers import (
    BEAM,
    DEBUG,
    NOOPT,
    Context,
    Profiling,
    Timing,
    ansilen,
    getenv,
)
from tinygrad.ops import Ops

if __name__ == "__main__":
    mdl = ResNet50()
    for p in nn.state.get_parameters(mdl):
        p.replace(Tensor.empty(p.shape))
    img = Tensor.empty(64, 3, 224, 224)

    PROFILE = getenv("PROFILE", 0)
    FORWARD_ONLY = getenv("FORWARD_ONLY", 0)
    SCHEDULE_ONLY = getenv("SCHEDULE_ONLY", 0)

    with Timing("all "):
        with Timing("***** model tensor in    "):
            out = mdl(img)

        if not FORWARD_ONLY:
            with Timing("***** model schedule in  "):

                # DEBUG: Inspect the output Tensor before calling .schedule() on it
                print(">>> out before .schedule():", out)

                sched = out.schedule()

                # DEBUG: Confirm if .schedule() returns the expected list of ScheduleItems
                print(">>> out.schedule call result:", sched)

            if not SCHEDULE_ONLY:
                asts = list(
                    {x.ast.key: x.ast for x in sched if x.ast.op is Ops.SINK}.values()
                )
                if (restrict_kernel := getenv("RESTRICT_KERNEL", -1)) != -1:
                    asts = asts[restrict_kernel : restrict_kernel + 1]
                kernels: List[Kernel] = []
                with Timing(f"***** model opts({len(asts):2d}) in  "):
                    with Profiling(PROFILE >= 3):
                        for ast in asts:
                            k = Kernel(ast)
                            if BEAM:
                                with Context(DEBUG=max(2, DEBUG.value)):
                                    k = beam_search(k, bufs_from_lin(k), BEAM.value)
                            elif NOOPT:
                                pass
                            else:
                                k = hand_coded_optimizations(k)
                            kernels.append(k)

                with Timing("***** model lower in     "):
                    uops = [
                        rewrite_shapetracker_with_index(k.get_optimized_ast(), k.opts)
                        for k in kernels
                    ]
                with Profiling(PROFILE, fn="/tmp/rewrite.prof"):
                    with Timing("***** model rewrite in   "):
                        rewritten_uops = []
                        for i, (k, u) in enumerate(zip(kernels, uops)):
                            with Timing(
                                f"rewrite {i:2d} {k.name}{' '*(50-ansilen(k.name))}",
                                enabled=getenv("VERBOSE", 0),
                            ):
                                rewritten_uops.append(full_graph_rewrite(u, k.opts))
                        uops = rewritten_uops
                if getenv("LINEARIZE", 1):
                    with Profiling(PROFILE >= 2):
                        with Timing("***** model linearize in "):
                            uops = [linearize_uop(u) for u in uops]
                    print(sum(len(u) for u in uops))
                    if getenv("SRC", 0):
                        renderer = Device[Device.DEFAULT].renderer
                        for k, u in zip(kernels, uops):
                            print(renderer.render(k.name, u))
