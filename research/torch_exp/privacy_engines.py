from opacus import PrivacyEngine as PrivacyEngineOpacus
from fastDP import PrivacyEngine as PrivacyEngineBK
from private_vision import PrivacyEngine as PrivacyEngineVision
from opacus.accountants.utils import get_noise_multiplier

# FastDP and private_vision libraries use a similar privacy engine. It will modify the internal methods for
# training, like step and backward.
# The privacy engine is returned, but it is actually never used, as the optimizer is attached to it.
# In the case of non private baseline, null is returned
def get_privacy_engine(model, loader, optimizer, lib, sample_rate, expected_batch_size, args):

    sigma = get_noise_multiplier(
        target_epsilon=args.epsilon,
        target_delta=args.target_delta,
        sample_rate=sample_rate,
        epochs=args.epochs,
        accountant=args.accountant,
    )

    print("Noise multiplier", sigma, flush=True)

    if lib == "fastDP":
        if "BK" in args.clipping_mode:
            clipping_mode = args.clipping_mode[3:]
        else:
            clipping_mode = "ghost"
        privacy_engine = PrivacyEngineBK(
            model,
            batch_size=expected_batch_size,
            sample_size=len(loader.dataset),
            noise_multiplier=sigma,
            epochs=args.epochs,
            clipping_mode=clipping_mode,
            origin_params=args.origin_params,
            accounting_mode=args.accountant,
        )
        privacy_engine.attach(optimizer)
        return privacy_engine

    elif lib == "private_vision":
        if "ghost" in args.clipping_mode:

            privacy_engine = PrivacyEngineVision(
                model,
                batch_size=expected_batch_size,
                sample_size=len(loader.dataset),
                noise_multiplier=sigma,
                epochs=args.epochs,
                max_grad_norm=args.grad_norm,
                ghost_clipping="non" not in args.clipping_mode,
                mixed="mixed" in args.clipping_mode,
            )
            privacy_engine.attach(optimizer)
            return privacy_engine

    return None


def get_privacy_engine_opacus(model, loader, optimizer, criterion, g, args):
    print("Opacus Engine")
    privacy_engine = PrivacyEngineOpacus(accountant=args.accountant)

    if args.clipping_mode == "O-ghost":
        model, optimizer, criterion, loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            epochs=args.epochs,
            target_epsilon=args.epsilon,
            target_delta=args.target_delta,
            max_grad_norm=args.grad_norm,
            criterion=criterion,
            grad_sample_mode="ghost",
            noise_generator=g,
        )
    else:
        model, optimizer, loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            epochs=args.epochs,
            target_epsilon=args.epsilon,
            target_delta=args.target_delta,
            max_grad_norm=args.grad_norm,
            noise_generator=g,
        )

    print(
        "optimizer params",
        "noise multiplier",
        optimizer.noise_multiplier,
        "max grad norm",
        optimizer.max_grad_norm,
        "loss reduction",
        optimizer.loss_reduction,
        "expected batch size",
        optimizer.expected_batch_size,
        flush=True,
    )

    return model, optimizer, loader, privacy_engine, criterion