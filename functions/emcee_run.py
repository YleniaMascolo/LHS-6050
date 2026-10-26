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
                       nproc=60,
                       check_interval=1000,
                       tol=0.03,
                       min_iter=1000,
                       resume=True,
                       fallback_burnin_pct=0.50):

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

        converged = False
        burnin_steps = 0
        sampling_target = niter

        for sample in sampler.sample(initial_state, iterations=niter, progress=False):
            iteration = sampler.iteration

            pct = int(100 * iteration / niter)  

            if pct != last_pct:   # print percentage
                print(f"Progress: {pct}% ({iteration}/{niter})")
                last_pct = pct

            # check convergence
            if not converged and iteration >= min_iter and iteration % check_interval == 0:
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

                    if np.all(delta_mean < tol) and np.all(delta_std < tol):
                        print(f"\n[CONVERGENCE REACHED] at step {iteration}!")
                        converged = True
                        burnin_steps = iteration  # pre-convergence step = burn-in
                        
                        # number of step AFTER convergence
                        sampling_target = iteration + max(30000, check_interval * 2)
                        print(f"Sampling will continue until step {sampling_target} to collect clean samples...")

                old_mean = new_mean
                old_std  = new_std

            if converged and iteration >= sampling_target:
                print("\n[DONE] Sampling phase completed!")
                break

    # ===== POST PROCESSING =====
    print("\nPost-processing...")

    # burnin = int(0.1 * sampler.iteration)  # 10% step
    if converged:
        burnin = burnin_steps 
        print(f"Status: Convergence reached at step {burnin}.")
    else:
        # manual burn-in
        burnin = int(fallback_burnin_pct * sampler.iteration)
        print(f"Status: Convergence NOT reached within limits. Applying manual fallback burn-in ({int(fallback_burnin_pct*100)}%).")
    thin = 1

    flat_samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)

    print(f"Tot steps: {sampler.iteration}")
    print(f"Burn-in (Pre-convergence): {burnin}")
    print(f"Pure Sampling steps: {sampler.iteration - burnin}")
    print(f"Final shape: {flat_samples.shape}")

    return sampler, flat_samples
