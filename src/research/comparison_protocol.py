"""Enforce a frozen comparison protocol without equating different algorithms."""

def validate_protocol(raw,settings,horizon,methods,train_seed,eval_seeds,journal=False,smoke=False,pilot=False):
    research=raw['research'];protocol=research.get('journal_protocol')
    if (journal or pilot) and not protocol:raise ValueError('No frozen journal protocol for this benchmark')
    record=dict(enforced=bool((journal or pilot) and not smoke),workability_only=bool(smoke),
                study='development' if smoke else ('pilot' if pilot else ('publication_candidate' if journal else 'development')),
                evaluation_metric='common likelihood-based sPCE in nats',
                inference_metric='secondary parameter MSE; not policy objective')
    if not protocol:return record
    record.update(version=protocol['version'],comparison=protocol['comparison'],
                  training_episode_cap=protocol.get('training_episode_cap',protocol.get('training_forward_cap')),
                  budget_unit=protocol['budget_unit'],checkpoint_selection=protocol['checkpoint_selection'],
                  claim_limit=protocol['claim_limit'],
                  method_specific_settings='Replay, batch sizes and optimizers may differ across algorithms; actual costs must be reported')
    if smoke and set(eval_seeds)&(set(protocol['evaluation_seeds'])-set(protocol['previously_used_evaluation_seeds'])):
        raise ValueError('Smoke must not consume reserved journal evaluation seeds')
    if not (journal or pilot) or smoke:return record
    if raw['system']['name']!='location_finding':raise ValueError('This frozen protocol is location-specific')
    if horizon!=30:raise ValueError('Journal location horizon is T30')
    if raw['observation']!={'noise_sigma':.5,'sampling':'normal_log_signal'}:raise ValueError('Changed location observation model')
    if raw['location_finding']['design_bounds']!=protocol.get('design_bounds',[-4.,4.]):raise ValueError('Changed design bounds')
    if raw['location_finding']['random_design_distribution']!=protocol.get('random_design_distribution','uniform_box'):raise ValueError('Changed random design distribution')
    if raw['research']['paper_protocol']['nominal_training_trajectories']!=protocol.get('training_episode_cap',protocol.get('training_forward_cap')):
        raise ValueError('Conflicting simulation caps')
    if train_seed not in protocol['training_seeds']:raise ValueError('Training seed is outside frozen journal seed list')
    if pilot:
        if train_seed!=101 or eval_seeds!=[1001]:raise ValueError('Pilot requires train101 and eval1001 only')
    elif sorted(eval_seeds)!=sorted(protocol['evaluation_seeds']):raise ValueError('Use all five declared evaluation seeds')
    record['previously_used_evaluation_seed_overlap']=sorted(set(eval_seeds)&set(protocol['previously_used_evaluation_seeds']))
    record['untouched_confirmation_data']=not bool(record['previously_used_evaluation_seed_overlap'])
    expected=research['settings']
    for key,value in expected.items():
        if settings[key]!=value:raise ValueError(f'Journal override changes frozen setting: {key}')
    for key,value in dict(training_bound=4,moe_revision=4,first_design_revision=2,spce_curriculum=1,
                          batch_size=research['settings']['batch_size'],contrasts=1024,eval_systems=2000,eval_contrasts=1000000).items():
        if settings[key]!=value:raise ValueError(f'Incorrect aligned setting: {key}')
    profile=research['upstream_vsoed']
    actual=profile['initial_replay']+profile['updates']*profile['new_trajectories']
    if actual!=protocol.get('training_episode_cap',protocol.get('training_forward_cap')):raise ValueError('vsOED must meet the exact common simulation cap')
    if not 0 < profile['discount'] <= 1.:raise ValueError('Training discount must be in (0,1]')
    record['vsOED_training_discount']=profile['discount']
    record['evaluation_discount']=1.0
    adaptive=research['training_profiles']['adaptive']
    if adaptive!=dict(learning_rate=.00005,spce_lr_decay=.98,spce_adam_beta1=.8,
                     spce_adam_beta2=.998,spce_lr_schedule_every=1000):
        raise ValueError('Changed frozen shared DAD/MoE optimizer')
    record['spce_learners']=['dad','moe_sboed','matched_dense','uniform_moe']
    record['vsOED_adaptations']=['exact simulation cap','method-specific training discount; common undiscounted final evaluation']
    return record
