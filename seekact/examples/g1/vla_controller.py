"""Bounded agent tool use after visual inspection and completed VLA preparation."""
import time


def wait(ctx,job_id,timeout):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        result=ctx.value('g1_vla_result',job_id=job_id)
        if 'completed' in result:return result
        time.sleep(.1)
    ctx.value('g1_vla',action='pause')
    raise RuntimeError('VLA result timeout; inspect current owner and scene')


def prepare(ctx):
    """Caller checks empty hands/site first; inspect a fresh image after return."""
    job=ctx.value('g1_vla',action='prepare')
    result=wait(ctx,job['job_id'],20)
    return dict(result=result,observation=ctx.value('g1_observe'))


def segment(ctx,task,steps,evidence):
    """Evidence is authored by the agent from its current visual observation."""
    job=ctx.value('g1_vla',action='run',task=task,steps=steps,evidence=evidence)
    result=wait(ctx,job['job_id'],steps/50+20)
    return dict(result=result,observation=ctx.value('g1_observe'),grasp_verified=False)
