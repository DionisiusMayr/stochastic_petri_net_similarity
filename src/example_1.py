from typing import Tuple

from pm4py import Marking
from pm4py.objects.petri_net.stochastic.obj import StochasticPetriNet

from src.similarity import ExplorationStrategy, logger, compute_similarity
from src.utils_net import simple_plot_and_save_net


def get_spn_1() -> Tuple[StochasticPetriNet, Marking]:
    spn = StochasticPetriNet()

    pi = StochasticPetriNet.Place("pi")
    pf = StochasticPetriNet.Place("pf")
    spn.places.add(pi)
    spn.places.add(pf)

    t1 = StochasticPetriNet.Transition(name="t1", label="A", weight=3)
    t2 = StochasticPetriNet.Transition(name="t2", label="B", weight=1)
    spn.transitions.add(t1)
    spn.transitions.add(t2)

    arcs = [
        (pi, t1),
        (pi, t2),
        (t1, pf),
        (t2, pf),
    ]

    for fr, to in arcs:
        arc = StochasticPetriNet.Arc(fr, to, weight=1)
        spn.arcs.add(arc)
        if isinstance(fr, StochasticPetriNet.Transition):
            fr.out_arcs.add(arc)
        else:
            to.in_arcs.add(arc)

    init_marking = Marking()
    init_marking[pi] += 1

    return spn, init_marking


def get_spn_2() -> Tuple[StochasticPetriNet, Marking]:
    spn = StochasticPetriNet()

    pi = StochasticPetriNet.Place("pi")
    pf = StochasticPetriNet.Place("pf")
    spn.places.add(pi)
    spn.places.add(pf)

    t1 = StochasticPetriNet.Transition(name="t1", label="A", weight=1)
    t2 = StochasticPetriNet.Transition(name="t2", label="B", weight=2)
    spn.transitions.add(t1)
    spn.transitions.add(t2)

    arcs = [
        (pi, t1),
        (pi, t2),
        (t1, pf),
        (t2, pf),
    ]

    for fr, to in arcs:
        arc = StochasticPetriNet.Arc(fr, to, weight=1)
        spn.arcs.add(arc)
        if isinstance(fr, StochasticPetriNet.Transition):
            fr.out_arcs.add(arc)
        else:
            to.in_arcs.add(arc)

    init_marking = Marking()
    init_marking[pi] += 1

    return spn, init_marking


if __name__ == '__main__':
    # Specify the hyperparameters according to your scenario
    compute_similarity.EXPLORATION_STRATEGY = ExplorationStrategy.QUEUE
    compute_similarity.MIN_TRACE_PROB = 1E-5
    compute_similarity.MAX_FIRING_SEQ_LENGTH = 10

    # Create two stochastic Petri nets in pm4py format
    spn_1, init_marking_1 = get_spn_1()
    spn_2, init_marking_2 = get_spn_2()

    # Optional: generate graphical visualization of them
    simple_plot_and_save_net(spn_1, init_marking_1, './net_1.png')
    simple_plot_and_save_net(spn_2, init_marking_2, './net_2.png')

    logger.info("Computing similarity by generating both stochastic languages...")
    logger.info(
        f"Parameters: "
        f"Exploration strategy: {compute_similarity.EXPLORATION_STRATEGY.name}, "
        f"max firing sequence length: {compute_similarity.MAX_FIRING_SEQ_LENGTH}, "
        f"min trace probability {compute_similarity.MIN_TRACE_PROB}"
    )
    lower_bound, upper_bound = compute_similarity(spn_1, init_marking_1, spn_2, init_marking_2)

    logger.info(f"Similarity bounds between SPN 1 and SPN 2 is: [{lower_bound:.2f}, {upper_bound:.2f}]")
