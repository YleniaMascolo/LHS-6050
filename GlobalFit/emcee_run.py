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

    ndim = len(init_guess[0])

    if nproc is None:
        nproc = max(1, (os.cpu_count() or 2) - 1)

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

            if pct != last_pct:   # print percentage
                print(f"Progress: {pct}% ({iteration}/{niter})")
                last_pct = pct

            if iteration % check_interval != 0 or iteration < min_iter:
                continue

            chain = sampler.get_chain(flat=False)
            flat_chain = chain.reshape(-1, ndim)

            new_mean = np.mean(flat_chain, axis=0)
            new_std  = np.std(flat_chain, axis=0)

            print(f"\nCheck step {iteration}")
            print("Mean:", new_mean)
            print("Std :", new_std)

            if old_mean is not None:

                delta_mean = np.abs(new_mean - old_mean) / (np.abs(old_mean) + 1e-8)
                delta_std  = np.abs(new_std - old_std) / (np.abs(old_std) + 1e-8)

                print("Δ mean:", delta_mean)
                print("Δ std :", delta_std)

                if (np.all(delta_mean < tol) and
                    np.all(delta_std < tol)):

                    print("\n DONE!")
                    break

            old_mean = new_mean
            old_std  = new_std

    # ===== POST PROCESSING =====

    print("\nPost-processing...")

    burnin = int(0.1 * sampler.iteration)  # 10% step
    thin = 1

    flat_samples = sampler.get_chain(discard=burnin,
                                     thin=thin,
                                     flat=True)

    print(f"Tot steps: {sampler.iteration}")
    print(f"Burn-in: {burnin}")
    print(f"Thin: {thin}")
    print(f"Final shape: {flat_samples.shape}")

    return sampler, flat_samples