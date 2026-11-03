"""Model-free stage planning for the published fixed-budget profile."""


def plan(c):
    if c.get('format_version') != 1:
        raise ValueError('Unsupported configuration version')
    return [dict(
        stage=name,
        epochs=c['stages'][name]['epochs'],
        parent=('public_pretrained_weights' if name == 'sft' else
                'initialized_SFT_weights' if name == 'preparation' else
                'preparation.final'),
        selection='final' if name == 'preparation' else 'validation_MAP_MSE_best',
    ) for name in ['sft', 'preparation', 'rl']]
