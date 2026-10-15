from src.research.sparse_moe import routing_gamma

def test_legacy_schedule_is_preserved():
    for x in [0.,.25,.5,.75,.8,1.]:
        assert routing_gamma(x)==max(.001,(1-min(x/.8,1))**2)

def test_delayed_schedule_has_warmup_and_terminal_sparse_phase():
    values=[routing_gamma(i/100,.3,.95) for i in range(101)]
    assert values[0]==values[30]==1.
    assert abs(values[95]-.001)<1e-12
    assert values[100]==values[95]
    assert all(a>=b for a,b in zip(values,values[1:]))
    assert routing_gamma(.75,.3,.95)>routing_gamma(.75)
