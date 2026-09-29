use solver::preflop::{
    BucketPolicy, PreflopConfig, PreflopSolver,
    equity::{EquityTable, NUM_CLASSES, class_prob},
};
use std::sync::Arc;
#[test]
fn conditional_action_values_match_closed_form_and_do_not_change_strategy() {
    let cfg: PreflopConfig = serde_json::from_value(serde_json::json!({
        "positions":["SB","BB"],"posts":[0.5,1.0],"stack":10.0,
        "limp":false,"open_raises":[2.0],"raise_mults":[],"max_raises":1,
        "add_allin":false,"realization":"raw","rake_pct":0.0
    }))
    .unwrap();
    let eq = Arc::new(EquityTable::build(1000));
    let mut s = PreflopSolver::new(cfg, eq.clone()).unwrap();
    let calls = BucketPolicy {
        call: vec![1.0; NUM_CLASSES],
        raise: vec![0.0; NUM_CLASSES],
        jam: vec![0.0; NUM_CLASSES],
        raise_size: "min".into(),
        raise_sizes: vec![],
        raise_multiples: vec![],
    };
    s.lock_point(&[1], Some(calls)).unwrap();
    let before = s.arena_snapshot();
    for br in [false, true] {
        let values = s.diagnostic_action_evs(&[], br).unwrap();
        let facing = s.diagnostic_action_evs(&[1], br).unwrap();
        for h in 0..NUM_CLASSES {
            let e: f64 = (0..NUM_CLASSES)
                .map(|j| eq.eq(h, j) as f64 * class_prob(j) as f64)
                .sum();
            assert!(values[0][h].abs() < 1e-6);
            assert!((values[1][h] - (4.0 * e - 1.5)).abs() < 2e-5);
            assert!(facing[0][h].abs() < 1e-6);
            assert!((facing[1][h] - (4.0 * e - 1.0)).abs() < 2e-5);
        }
    }
    assert_eq!(before, s.arena_snapshot());
    assert!(s.diagnostic_action_evs(&[999], false).is_err());
}

#[test]
fn node_view_exports_action_major_evs_only_with_a_published_strategy() {
    let cfg: PreflopConfig = serde_json::from_value(serde_json::json!({
        "positions":["SB","BB"],"posts":[0.5,1.0],"stack":10.0,
        "limp":false,"open_raises":[2.0],"raise_mults":[],"max_raises":1,
        "add_allin":false,"realization":"raw","rake_pct":0.0
    }))
    .unwrap();
    let mut s = PreflopSolver::new(cfg, Arc::new(EquityTable::build(1))).unwrap();
    // Unsolved: the strategy is withheld, and so are the values.
    let unsolved = s.node_view(&[]).unwrap();
    assert!(unsolved.strategy_note.is_some());
    assert!(unsolved.strategy.is_none() && unsolved.evs.is_none());
    for _ in 0..50 {
        s.iterate();
    }
    for path in [vec![], vec![1]] {
        let view = s.node_view(&path).unwrap();
        assert_eq!(view.kind, "action");
        let na = view.actions.len();
        let evs = view.evs.as_ref().expect("solved action node exports evs");
        assert_eq!(evs.len(), na * NUM_CLASSES);
        assert_eq!(view.strategy.as_ref().unwrap().len(), evs.len());
        let expected = s.diagnostic_action_evs(&path, false).unwrap();
        let fold = view.actions.iter().position(|a| a.kind == "fold").unwrap();
        for a in 0..na {
            for h in 0..NUM_CLASSES {
                let got = evs[a * NUM_CLASSES + h];
                assert!((got as f64 - expected[a][h]).abs() < 1e-4, "{path:?} a={a} h={h}");
            }
        }
        assert!(evs[fold * NUM_CLASSES..(fold + 1) * NUM_CLASSES].iter().all(|v| *v == 0.0));
        // A non-fold column carries real hand-dependent values.
        let other = (fold + 1) % na;
        let col = &evs[other * NUM_CLASSES..(other + 1) * NUM_CLASSES];
        assert!(col.iter().any(|v| (v - col[0]).abs() > 1e-3));
    }
    // SB folds: a fold_win terminal has no actor, hence no values in the JSON.
    let fold = s.node_view(&[]).unwrap().actions.iter().position(|a| a.kind == "fold").unwrap();
    let terminal = s.node_view(&[fold]).unwrap();
    assert_ne!(terminal.kind, "action");
    assert!(terminal.evs.is_none());
    assert!(serde_json::to_value(&terminal).unwrap().get("evs").is_none());
}
