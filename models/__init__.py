"""Explicit model selection: legacy models and the pinned official backbone."""
from .classifier import build_model as build_legacy_model


def build_model(config):
    if config['model']['backbone'] == 'official_pointmamba':
        from .official_pointmamba import OfficialPointMamba
        options = dict(config['model'])
        options.pop('backbone')
        return OfficialPointMamba(config['data']['num_classes'], **options)
    return build_legacy_model(config)
