import torch, os

def test_nccl_ops():
    num_gpu = torch.cuda.device_count()

    if num_gpu <= 1:
        return
    import torch.multiprocessing as mp

    dist_url = "tcp://127.0.0.1:9500"
    mp.spawn(_test_nccl_worker, nprocs=num_gpu, args=(num_gpu, dist_url), daemon=False)
    print("NCCL init succeeded.")


def _test_nccl_worker(rank, num_gpu, dist_url):
    import torch.distributed as dist

    dist.init_process_group(backend="NCCL", init_method=dist_url, rank=rank, world_size=num_gpu)
    dist.barrier()

test_nccl_ops()
