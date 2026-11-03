"""Reject declarations that disagree with this finite entry's executed constants."""


def validate_frozen_training_config(config):
    if config.get('dgf_complete') or config.get('matched_dgf_baseline'):
        from .dgf_reference_rl import validate_contract
        return validate_contract(config)
    phase = config['phase']
    if phase not in ('preparation', 'scientific'):
        raise ValueError('This training entry supports preparation or scientific phase only')
    preparation = phase == 'preparation'
    batch = config.get('global_input_batch', 512)
    if not isinstance(batch, int) or isinstance(batch, bool) or batch < 1:
        raise ValueError('Independent input batch must be a positive integer')
    if batch != 512 and not config.get('qualification_only'):
        raise ValueError('Formal independent input batch is frozen to 512')
    campaign = bool(config.get('warmup_campaign'))
    requested_epochs = int(config.get('epochs', 1 if preparation else 4))
    requested_updates = int(config.get('updates_per_input_batch', 1 if preparation else 2))
    maximum = 16 if config.get('four_horizon_campaign') else 4
    if campaign and not preparation and not 1 <= requested_epochs <= maximum:
        raise ValueError('Scientific epochs exceed the registered campaign limit')
    supported = dict(J=8, G=4, K=1 if config['route'] == 'A0' else 8,
        epochs=requested_epochs if campaign else (1 if preparation else 4),
        updates_per_input_batch=requested_updates if campaign else (1 if preparation else 2),
        PPO_clip=.2, reference_beta=.01, gradient_norm_clip=1., schedule='constant',
        global_input_batch=batch)
    if config.get('four_horizon_campaign') and config.get('continuous_recipe')=='R09':
        supported['G']=8
    point = (str(config.get('scientific_path', '')).startswith('point-')
        or str(config.get('objective_version', '')).startswith('point-opd-')
        or config.get('point_selection_only') is True)
    if point:
        supported.update(force_fixed_kappa=True, kappa=None if config['route'] == 'A0' else 8., credit='position')
    arm = config.get('algorithm_arm', config.get('arm'))
    if not preparation and arm in ('E16','E17'):
        supported['nll_anchor_coefficient'] = 0.
    for key, actual in supported.items():
        if key not in config:
            continue
        declared = config[key]
        equal = declared == actual
        if isinstance(declared, bool) != isinstance(actual, bool):
            equal = False
        if not equal:
            raise ValueError(f'{key} declares {declared!r}, but this finite entry executes {actual!r}')
    return dict(supported, phase=phase, J_G_scope='Native scientific RL only; not OPD block MC samples',
                native_RL_sampling_active=not preparation and config.get('algorithm_arm',config.get('arm')) not in ('E00','E02'))
