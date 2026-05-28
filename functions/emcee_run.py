import multiprocessing as mp
import emcee
import os 
import numpy as np


def emcee_adaptive_run(init_guess,
                       nwalkers,
                       niter,
                       lnlike,
                       args,
                       filename,
                       nproc=None,
                       check_interval=1000,
                       tol=0.03,
                       min_iter=1000,
                       resume=True):
    """Runs an MCMC sampling using emcee with adaptive convergence monitoring

    Args:
        init_guess (ndarray): Initial position matrix for the walkers, 
            with shape (nwalkers, ndim).
        nwalkers (int): Number of walkers in the ensemble.
        niter (int): Maximum number of MCMC production steps to run.
        lnlike (callable): The log-likelihood function to sample.
        args (tuple/list): Additional positional arguments passed to `lnlike`.
        filename (str): Path to the HDF5 file used to store the chains backend.
        nproc (int, optional): Number of CPU processes for multiprocessing. 
            If None, defaults to (CPU_count - 1). Defaults to None.
        check_interval (int, optional): Step frequency at which convergence 
            is evaluated. Defaults to 1000.
        tol (float, optional): Fractional tolerance threshold for stopping. 
            Early stop triggers when both mean and std vary by less than `tol` 
            between consecutive checks. Defaults to 0.03.
        min_iter (int, optional): Minimum number of iterations required before 
            checking convergence. Defaults to 1000.
        resume (bool, optional): If True, attempts to resume the MCMC from the 
            existing HDF5 file if it exists. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - sampler (emcee.EnsembleSampler): The trained emcee sampler object.
            - flat_samples (ndarray): The flattened posterior samples array, 
              post-processed to discard the first 10% as burn-in.
    """
    ndim = len(init_guess[0])

    if nproc is None:
        # Fallback to 1 process if CPU count cannot be determined
        nproc = max(1, (os.cpu_count() or 2) - 1)

    # Initialize HDF5 backend for storage and resuming capabilities
    backend = emcee.backends.HDFBackend(filename)

    if resume and os.path.exists(filename):
        print("Resume chain...")
        initial_state = None
    else:
        print("New chain...")
        backend.reset(nwalkers, ndim)
        initial_state = init_guess

    old_mean = None
    old_std = None

    # Use 'spawn' context for robust multiprocessing across platforms
    with mp.get_context("spawn").Pool(processes=nproc) as pool:

        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, lnlike,
            args=args,
            pool=pool,
            backend=backend
        )

        last_pct = -1
        for sample in sampler.sample(initial_state,
                                     iterations=niter,
                                     progress=False):

            iteration = sampler.iteration
            pct = int(100 * iteration / niter)  

            # Progress monitoring
            if pct != last_pct:   
                print(f"Progress: {pct}% ({iteration}/{niter})")
                last_pct = pct

            # Check convergence only at specified intervals after min_iter
            if iteration % check_interval != 0 or iteration < min_iter:
                continue

            chain = sampler.get_chain(flat=False)
            flat_chain = chain.reshape(-1, ndim)

            new_mean = np.mean(flat_chain, axis=0)
            new_std  = np.std(flat_chain, axis=0)

            print(f"\nCheck step {iteration}")
            print("Mean:", new_mean)
            print("Std :", new_std)

            # Convergence criteria assessment
            if old_mean is not None:
                delta_mean = np.abs(new_mean - old_mean) / (np.abs(old_mean) + 1e-8)
                delta_std  = np.abs(new_std - old_std) / (np.abs(old_std) + 1e-8)

                print("Δ mean:", delta_mean)
                print("Δ std :", delta_std)

                # Early stop if all parameters meet the tolerance criteria
                if (np.all(delta_mean < tol) and
                    np.all(delta_std < tol)):

                    print("\n DONE!")
                    break

            old_mean = new_mean
            old_std  = new_std

    # ===== POST PROCESSING =====
    print("\nPost-processing...")

    # Default burn-in set to 10% of the completed iterations
    burnin = int(0.1 * sampler.iteration)  
    thin = 1

    flat_samples = sampler.get_chain(discard=burnin,
                                     thin=thin,
                                     flat=True)

    print(f"Tot steps: {sampler.iteration}")
    print(f"Burn-in: {burnin}")
    print(f"Thin: {thin}")
    print(f"Final shape: {flat_samples.shape}")

    return sampler, flat_samples