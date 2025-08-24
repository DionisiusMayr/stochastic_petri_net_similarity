import collections
import functools
import heapq
import time
from enum import Enum
from typing import Tuple, Dict, Union, Deque

from pm4py import Marking
from pm4py.objects.petri_net.stochastic.obj import StochasticPetriNet

from logger import setup_logger

# Set up logger for debugging and performance tracking
logger = setup_logger(__name__, 'spn_similarity.log')

# Maximum number of unique cached calls for LRU cache functions
CACHE_MAXSIZE = 5_000_000  # Large enough to cache frequent state checks efficiently


class ExplorationStrategy(Enum):
    """
    Enumeration of supported exploration strategies for traversal.
    - QUEUE: Breadth-first search
    - STACK: Depth-first search
    - HEAP: Best-first search (based on probability)
    """
    QUEUE = 0
    STACK = 1
    HEAP = 2


# Custom type aliases for better readability
Container = Union[list, Deque]  # Container to hold states (list, deque, or heap)
State = Tuple[float, frozenset, tuple, tuple]  # (probability, marking, trace, firing sequence)
StochasticLanguage = Dict[str, float]

class _TransitionChecker:
    """
    Helper class to efficiently get enabled transitions and manage transition firing
    for a given stochastic Petri Net.
    Utilizes caching to improve performance in large state spaces.
    """

    def __init__(self, net: StochasticPetriNet):
        """
        Initializes the Transition Checker for the given Petri net.
        Precomputes a mapping from each place to the transitions it can enable.
        """
        self.net = net
        self.transitions_from_place = collections.defaultdict(set)

        # Build a mapping from places to transitions they connect to
        for a in net.arcs:
            if isinstance(a.target, StochasticPetriNet.Transition):
                self.transitions_from_place[a.source.name].add(a.target)

        # Clear caches to avoid stale data issues
        _TransitionChecker.get_enabled_transitions.cache_clear()
        _TransitionChecker.execute.cache_clear()

    @functools.lru_cache(maxsize=CACHE_MAXSIZE)
    def get_enabled_transitions(self, m: frozenset):
        """
        Returns the set of enabled transitions for the given marking and
        the total weight of these transitions.
        Uses LRU cache to speed up repeated state checks.
        """
        _m = dict(m)
        enabled = set()
        for p in _m:
            for t in self.transitions_from_place[p]:
                add = True
                for a in t.in_arcs:
                    if _m.get(a.source.name, 0) < a.weight:
                        add = False
                        break
                if add:
                    enabled.add(t)

        return enabled, sum([t.weight for t in enabled]) if enabled else 0

    @functools.lru_cache(maxsize=CACHE_MAXSIZE)
    def execute(self, t: StochasticPetriNet.Transition, m: frozenset) -> frozenset:
        """
        Fires a transition on the given marking without checking if it's enabled.
        This is an unsafe execution for performance reasons.
        Returns the resulting marking as a frozenset for faster hashing.
        """
        m_out = dict(m)

        # Remove tokens from input places
        for a in t.in_arcs:
            m_out[a.source.name] = m_out.get(a.source.name, 0) - a.weight
            if m_out[a.source.name] <= 0:
                del m_out[a.source.name]

        # Add tokens to output places
        for a in t.out_arcs:
            m_out[a.target.name] = m_out.get(a.target.name, 0) + a.weight

        return frozenset(m_out.items())


def _init_to_visit(initial_state: State) -> Container:
    """
    Initializes the container that holds states to explore,
    based on the global exploration strategy.
    - QUEUE: deque for BFS
    - STACK: list for DFS
    - HEAP: list (used with heapq) for Best-first search
    """
    if compute_similarity.EXPLORATION_STRATEGY is ExplorationStrategy.QUEUE:
        return collections.deque([initial_state])
    elif compute_similarity.EXPLORATION_STRATEGY in (ExplorationStrategy.STACK, ExplorationStrategy.HEAP):
        return [initial_state]
    else:
        raise ValueError("Invalid exploration strategy")


def _pop(container: Container) -> State:
    """
    Pops the next state from the container based on the exploration strategy:
    - QUEUE: first element (FIFO)
    - STACK: last element (LIFO)
    - HEAP: most probable element using heapq
    """
    if compute_similarity.EXPLORATION_STRATEGY is ExplorationStrategy.HEAP:
        return heapq.heappop(container)
    elif compute_similarity.EXPLORATION_STRATEGY is ExplorationStrategy.QUEUE:
        return container.popleft()
    elif compute_similarity.EXPLORATION_STRATEGY is ExplorationStrategy.STACK:
        return container.pop()
    else:
        raise ValueError("Invalid exploration strategy")


def _push(container: Container, state: State) -> None:
    """
    Pushes a new state to the container based on the exploration strategy:
    - QUEUE/STACK: appends to list/deque
    - HEAP: uses heapq
    """
    if compute_similarity.EXPLORATION_STRATEGY is ExplorationStrategy.HEAP:
        heapq.heappush(container, state)
    elif compute_similarity.EXPLORATION_STRATEGY in (ExplorationStrategy.STACK, ExplorationStrategy.QUEUE):
        container.append(state)
    else:
        raise ValueError("Invalid exploration strategy")


def get_stochastic_language(
        net: StochasticPetriNet,
        initial_marking: Marking
) -> StochasticLanguage:
    """
    Explores and generates the stochastic language of the given Petri net starting
    from its initial marking.
    Traverses reachable markings and computes all possible traces with their probabilities.
    Returns:
        A dictionary where keys are traces (tuple of labels) and values are probabilities.
    """
    slang = collections.defaultdict(float)

    # Initial state: (probability, marking, trace, firing sequence)
    initial_state = (-1.0, frozenset({k.name: v for k, v in initial_marking.items()}.items()), (), ())
    to_visit = _init_to_visit(initial_state)
    visited = set()
    tc = _TransitionChecker(net)

    while len(to_visit) > 0:
        state = _pop(to_visit)
        prob, m, trace, firing_seq = state

        en_t, total_weight = tc.get_enabled_transitions(m)

        # Deadlock: no enabled transitions
        if not en_t:
            slang[trace] += -prob  # Negative because heap pops smallest element first

        for t in en_t:
            new_m = tc.execute(t, m)

            # Add transition label to trace (if available)
            new_trace = trace + (t.label,) if t.label is not None else trace
            new_firing_seq = firing_seq + (t.name,)

            if total_weight == 0:
                continue

            new_prob = prob * t.weight / total_weight
            new_state = (new_prob, new_m, new_trace, new_firing_seq)

            # Apply pruning rules to limit state space explosion
            if ((len(new_firing_seq) > compute_similarity.MAX_FIRING_SEQ_LENGTH)
                    or (-new_prob < compute_similarity.MIN_TRACE_PROB)
                    or ((new_m, new_firing_seq) in visited)):
                continue

            visited.add((new_m, new_firing_seq))
            _push(to_visit, new_state)

    return slang


def compute_similarity(
        m1_net: StochasticPetriNet,
        m1_im: Marking,
        m2_net: StochasticPetriNet,
        m2_im: Marking
) -> Tuple[float, float]:
    """
    Computes similarity bounds between two stochastic Petri nets:
    - Builds stochastic languages for both models.
    - Calculates lower bound (shared trace probabilities).
    - Calculates upper bound (max possible similarity).
    Returns:
        [lower_bound, upper_bound] as floats.
    """
    # Compute stochastic language of first net
    start1 = time.time()
    l1 = get_stochastic_language(m1_net, m1_im)
    end1 = time.time()
    explore_slanguage_1_time = end1 - start1
    logger.info(f"Took {explore_slanguage_1_time:.2f} seconds to compute L1")
    logger.info(f"L1 stochastic language: {l1}")

    # Compute stochastic language of second net
    start2 = time.time()
    l2 = get_stochastic_language(m2_net, m2_im)
    end2 = time.time()
    explore_slanguage_2_time = end2 - start2
    logger.info(f"Took {explore_slanguage_2_time:.2f} seconds to compute L2")
    logger.info(f"L2 stochastic language: {l2}")

    # Compute lower and upper similarity bounds
    lbound = 0.0
    ubound1 = 1.0
    ubound2 = 1.0
    for trace in l1:
        lbound += min(l1[trace], l2[trace])
        if l1[trace] > l2[trace]:
            ubound1 -= l1[trace] - l2[trace]
        else:
            ubound2 -= l2[trace] - l1[trace]

    ubound = min(ubound1, ubound2)

    logger.info(f"Total stochastic Language explored of M1: {sum(l1.values()):.2f}")
    logger.info(f"Total stochastic Language explored of M2: {sum(l2.values()):.2f}")

    return lbound, ubound


# Default hyperparameters for compute_similarity function
compute_similarity.EXPLORATION_STRATEGY = ExplorationStrategy.QUEUE  # BFS by default
compute_similarity.MIN_TRACE_PROB = 1E-8  # Ignore traces with very low probability
compute_similarity.MAX_FIRING_SEQ_LENGTH = 15  # Limit for firing sequence length
