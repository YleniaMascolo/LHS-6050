import os
import multiprocessing as mp
import numpy as np
from dynesty import NestedSampler



def dynesty_dynamic_run(ptform,
                         lnlike,
                         ndim,
                         args=(),
                         ptform_args=None,
                         filename="dynesty_checkpoint.pkl",
                         nproc=8,
                         nlive=1000,
                         dlogz_threshold=0.01,
                         resume=True):
    """
    Executes a Dynamic Nested Sampling run using dynesty.
    
    Parameters:
    -----------
    ptform : callable
        Prior transform function: ptform(u) -> p
    lnlike : callable
        Log-likelihood function: lnlike(p, *args)
    ndim : int
        Number of dimensions (parameters)
    args : tuple
        Extra arguments for lnlike
    filename : str
        Path for the checkpoint file (save/resume)
    nproc : int
        Number of processes for multiprocessing
    nlive : int
        Initial number of live points
    dlogz_threshold : float
        Stopping criterion for Evidence acquisition (default 0.01)
    resume : bool
        If True, attempts to resume from an existing checkpoint
    """

    if nproc is None:
        nproc = max(1, (os.cpu_count() or 2) - 1)

    sampler = None

    ctx = mp.get_context("spawn")
    
    with ctx.Pool(processes=nproc) as pool:

        # ===== CHECKPOINT / RESUME MANAGEMENT =====
        if resume and os.path.exists(filename):
            print(f"Resuming sampling from {filename}...")
            try:
                sampler = NestedSampler.restore(filename)
                print("Checkpoint successfully loaded!")
            except Exception as e:
                print(f"Could not load checkpoint ({e}). Starting a new chain...")

        # If no checkpoint exists or restoration failed, initialize the sampler
        if sampler is None:
            print("Starting a new chain with Dynesty...")
            
            # Initialize the sampler inside the checkpoint manager
            sampler = NestedSampler(
                lnlike,
                ptform,
                ndim,
                logl_args=args,
                ptform_args=ptform_args, # Send the circular flag to prior_transform
                nlive=nlive,
                # sample='rslice',
                pool=pool,
                queue_size=nproc
            )

        # ===== RUN SAMPLING =====
        print(f"Starting run (dlogz stopping threshold: {dlogz_threshold})...")
        
        # dynesty natively handles convergence criteria and checkpoints via run_nested()
        sampler.run_nested(dlogz=dlogz_threshold)

        # # Safely close the multiprocessing pool if it belongs to this scope
        # try:
        #     sampler.pool.close()
        #     sampler.pool.join()
        # except AttributeError:
        #     pass

    # ===== POST-PROCESSING =====
    print("\nExtracting results...")
    results = sampler.results


    weights = np.exp(results.logwt - results.logz[-1])
    weights /= np.sum(weights)
    
    from dynesty.utils import resample_equal
    flat_samples = resample_equal(results.samples, weights)

    print(f"Final Log-Evidence (ln Z): {results.logz[-1]:.3f} +/- {results.logzerr[-1]:.3f}")
    print(f"Total number of samples generated: {len(results.samples)}")
    print(f"Final posterior shape: {flat_samples.shape}")

    return results, flat_samples

