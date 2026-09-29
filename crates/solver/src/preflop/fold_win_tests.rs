//! Fold-out rake gate (`PreflopConfig::fold_win_min_voluntary`) checked
//! through the real terminal payoffs the CFR traversal consumes.

use super::*;
use std::sync::LazyLock;

static TABLE: LazyLock<Arc<EquityTable>> = LazyLock::new(|| Arc::new(EquityTable::build(8)));

fn table() -> Arc<EquityTable> {
    TABLE.clone()
}

/// BTN/SB/BB, 20 bb, opens to 3, one 3-bet (to 9), 5% uncapped rake that
/// also applies to fold-outs.
fn raked_fold_outs(min_voluntary: u8) -> PreflopSolver {
    let cfg: PreflopConfig = serde_json::from_value(serde_json::json!({
        "positions": ["BTN", "SB", "BB"], "posts": [0.0, 0.5, 1.0], "stack": 20.0,
        "limp": false, "open_raises": [3.0], "raise_mults": [3.0], "max_raises": 2,
        "add_allin": false, "rake_pct": 5.0, "rake_cap": 0.0, "no_flop_no_drop": false,
        "fold_win_min_voluntary": min_voluntary, "realization": "raw"
    }))
    .unwrap();
    PreflopSolver::new(cfg, table()).unwrap()
}

fn fold_win(s: &PreflopSolver, invested: [f64; 3]) -> usize {
    s.nodes
        .iter()
        .position(|n| {
            n.kind == KIND_FOLD_WIN
                && n.invested.iter().zip(invested).all(|(a, b)| (a - b).abs() < 1e-9)
        })
        .expect("fold-out terminal")
}

/// Normalized ranges, so every terminal's reach product is exactly 1.
fn unit_reaches(n: usize) -> Vec<Vec<f32>> {
    (0..n).map(|_| (0..NUM_CLASSES).map(class_prob).collect()).collect()
}

/// Rake actually charged at a fold-out, recovered from the winner's payoff.
fn charged_rake(s: &PreflopSolver, node: usize) -> f64 {
    let nd = &s.nodes[node];
    let w = nd.winner as usize;
    let mut out = [0f32; NUM_CLASSES];
    s.terminal_value(node, w, &unit_reaches(s.n), &mut out);
    nd.pot - nd.invested[w] - out[0] as f64
}

#[test]
fn one_voluntary_fold_out_is_raked_only_below_the_gate() {
    for (min, open_rake, walk_rake) in [(1u8, 0.125, 0.05), (2, 0.0, 0.0)] {
        let s = raked_fold_outs(min);
        // BTN opens to 3, both blinds fold: matched pot 2.5 at 5%.
        let open = fold_win(&s, [3.0, 0.5, 1.0]);
        assert!((charged_rake(&s, open) - open_rake).abs() < 1e-5, "min {min}: open-fold rake");
        // Walk (no voluntary player): the legacy rake on the blinds stays at 1.
        let walk = fold_win(&s, [0.0, 0.5, 1.0]);
        assert!((charged_rake(&s, walk) - walk_rake).abs() < 1e-5, "min {min}: walk rake");
    }
}

#[test]
fn two_voluntary_fold_out_is_raked_under_both_gates() {
    for min in [1u8, 2] {
        let s = raked_fold_outs(min);
        // BTN opens to 3, SB 3-bets to 9, BB folds, BTN folds: matched pot 7.
        let three_bet = fold_win(&s, [3.0, 9.0, 1.0]);
        assert_eq!(s.nodes[three_bet].winner, 1);
        assert!((charged_rake(&s, three_bet) - 0.35).abs() < 1e-5, "min {min}");
    }
}

#[test]
fn every_terminal_conserves_chips_net_of_rake_under_both_gates() {
    for min in [1u8, 2] {
        let s = raked_fold_outs(min);
        let weights: Vec<Vec<f32>> = (0..3)
            .map(|p| (0..NUM_CLASSES).map(|h| class_prob(h) * ((h + p * 17) % 23 + 1) as f32 / 25.0).collect())
            .collect();
        let probability: f64 = weights.iter().map(|w| w.iter().sum::<f32>() as f64).product();
        let mut checked = 0;
        for (node, nd) in s.nodes.iter().enumerate().filter(|(_, nd)| nd.kind != KIND_ACTION) {
            let mut total = 0.0;
            for p in 0..3 {
                let mut values = [0f32; NUM_CLASSES];
                s.terminal_value(node, p, &weights, &mut values);
                total += values.iter().zip(&weights[p]).map(|(v, w)| *v as f64 * *w as f64).sum::<f64>();
            }
            let rake = if nd.kind == KIND_FOLD_WIN { s.fold_win_rake(nd) } else { s.rake_of(nd.pot) };
            // Fold-outs pay constant deltas (exact to f32 rounding); pot
            // shares sum f32 class equities, so allow 1e-4 of the pot mass.
            let tolerance = if nd.kind == KIND_FOLD_WIN { 1e-6 } else { 1e-4 * probability * nd.pot };
            assert!(
                (total + probability * rake).abs() < tolerance,
                "min {min}, node {node}: payoffs {total} + rake {rake} x {probability}"
            );
            checked += 1;
        }
        assert!(checked > 5, "min {min}: only {checked} terminals");
    }
}

#[test]
fn gate_outside_one_to_seat_count_is_rejected() {
    for bad in [0u8, 4] {
        let mut cfg = raked_fold_outs(1).cfg.clone();
        cfg.fold_win_min_voluntary = bad;
        let err = PreflopSolver::new(cfg, table()).err().expect("invalid gate must be rejected");
        assert!(err.contains("fold_win_min_voluntary must be in 1..=3"), "{err}");
    }
    assert!(PreflopSolver::new(raked_fold_outs(3).cfg.clone(), table()).is_ok());
}
