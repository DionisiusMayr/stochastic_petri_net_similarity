import tempfile
from argparse import ArgumentError
from typing import List, Tuple, Dict

import numpy as np
import pm4py
from graphviz import Digraph
from pm4py import Marking
from pm4py.objects.petri_net.stochastic.obj import StochasticPetriNet
from pm4py.visualization.petri_net import visualizer as pn_visualizer


def to_SPN(net_data: Dict[str, List], initial_marking: List[str], final_marking: List[str]) -> Tuple[
    StochasticPetriNet, pm4py.Marking, pm4py.Marking]:
    """
    Creates a StochasticPetriNet based on a simplified description of one and its initial/final markings.

    @param net_data: The simplified description of a Stochastic Petri net as a dictionary.
    @param initial_marking: The initial marking of the Petri net.
    @param final_marking: The final marking of the Petri net.
    @returns: a pm4py.StochasticPetriNet object, a pm4py.Marking object representing the initial marking of the net and
    a pm4py.Marking object representing the final marking.

    Example of a net_data representation:
    {
        'P': ['p_i', 'p_f'],
        'T': [['A', 80], ['B', 'new_label_for_B', 20]],  # The 80 and 20 here are the weights of the transitions.
                                                         # Each list of 'T' can have either 2 or 3 arguments, which
                                                         # helps to create tau transitions (use `None` as the label).
        'F': [["p_i", "A"],
              ["p_i", "B"],
              ["A", "p_f"],
              ["B", "p_f"]]
    }

    Observations:
    - Any transition starting with "tau" will have no label.
    """
    net = StochasticPetriNet()

    # These two dictionaries are used to allow us to reference the same places/transitions when creating arcs or markings.
    places = {p: StochasticPetriNet.Place(p) for p in net_data['P']}
    transitions = {}
    for d in net_data['T']:
        if len(d) == 2:
            t = d[0]
            n = t
            w = d[1]
        elif len(d) == 3:
            n = d[0]
            t = d[1]
            w = d[2]
        else:
            raise ArgumentError

        label = t if not t.startswith('tau') else None
        transitions[n] = StochasticPetriNet.Transition(name=n, label=label, weight=w)

    for p in places:
        net.places.add(places[p])

    for t in transitions:
        net.transitions.add(transitions[t])

    for f in net_data['F']:
        if f[0] in places:
            fr = places[f[0]]
            to = transitions[f[1]]
        else:
            fr = transitions[f[0]]
            to = places[f[1]]

        # This has the same functionality as the pm4py function `add_arc_from_to`.
        a = StochasticPetriNet.Arc(fr, to, weight=1)
        net.arcs.add(a)
        fr.out_arcs.add(a)
        to.in_arcs.add(a)

    ret_initial_marking = pm4py.Marking()
    for im in initial_marking:
        ret_initial_marking[places[im]] += 1

    ret_final_marking = pm4py.Marking()
    for fm in final_marking:
        ret_final_marking[places[fm]] += 1

    return net, ret_initial_marking, ret_final_marking

def _graphviz_visualization(net, image_format="png", initial_marking=None, final_marking=None, decorations=None,
                            font_size="12"):
    # Customized version of the function from pm4py
    if decorations is None:
        decorations = {}

    font_size = str(font_size)

    filename = tempfile.NamedTemporaryFile(suffix='.gv')
    filename.close()

    viz = Digraph(net.name, filename=filename.name, engine='dot', graph_attr={'bgcolor': 'white'})
    viz.graph_attr['rankdir'] = 'LR'
    viz.graph_attr['dpi'] = '200'

    # transitions
    viz.attr('node', shape='box')
    for t in net.transitions:
        label = decorations[t]["label"] if t in decorations and "label" in decorations[t] else ""
        fillcolor = decorations[t]["color"] if t in decorations and "color" in decorations[t] else None
        textcolor = "black"

        if t.label is not None and not label:
            label = t.label

        label = str(label)

        if fillcolor is None:
            if t.label is None:
                fillcolor = "black"
                textcolor = "white"
                if not label:
                    label = "τ"
            else:
                fillcolor = 'white'

        if t.name in decorations and "weight" in decorations[t.name]:
            viz.node(str(id(t)), label + '\n' + decorations[t.name]['weight'], style='filled', fillcolor=fillcolor,
                     border='1', fontsize=font_size, fontcolor=textcolor)
        else:
            viz.node(str(id(t)), label, style='filled', fillcolor=fillcolor, border='1', fontsize=font_size,
                     fontcolor=textcolor)

    # places
    # add places, in order by their (unique) name, to avoid undeterminism in the visualization
    places_sort_list_im = sorted([x for x in list(net.places) if x in initial_marking], key=lambda x: x.name)
    places_sort_list_fm = sorted([x for x in list(net.places) if x in final_marking and not x in initial_marking],
                                 key=lambda x: x.name)
    places_sort_list_not_im_fm = sorted(
        [x for x in list(net.places) if x not in initial_marking and x not in final_marking], key=lambda x: x.name)
    # making the addition happen in this order:
    # - first, the places belonging to the initial marking
    # - after, the places not belonging neither to the initial marking and the final marking
    # - at last, the places belonging to the final marking (but not to the initial marking)
    # in this way, is more probable that the initial marking is on the left and the final on the right
    places_sort_list = places_sort_list_im + places_sort_list_not_im_fm + places_sort_list_fm

    for p in places_sort_list:
        label = decorations[p.name]["label"] if p.name in decorations and "label" in decorations[p.name] else ""
        fillcolor = decorations[p.name]["color"] if p.name in decorations and "color" in decorations[
            p.name] else 'white'

        label = str(label)
        if p in initial_marking:
            if initial_marking[p] == 1:
                if p.name in decorations and "label" in decorations[p.name]:
                    with viz.subgraph() as s:
                        s.attr(rank='same', rankdir='TB')
                        s.node(str(id(p)), "<&#9679;>", fontsize='34', fixedsize='true', shape="circle", width='0.75',
                               style="filled", fillcolor=fillcolor)
                        _name = f"label_{str(id(p))}"
                        s.node(_name, decorations[p.name]['label'], shape='plaintext', fontsize='14', fixedsize='true',
                               width='0', height='0', margin='0')
                        s.edge(str(id(p)), _name, style='invis', weight='100', minlen='0')
                else:
                    viz.node(str(id(p)), "<&#9679;>", fontsize="34", fixedsize='true', shape="circle", width='0.75',
                             style="filled", fillcolor=fillcolor)
            else:
                marking_label = str(initial_marking[p])
                if len(marking_label) >= 3:
                    if p.name in decorations and "label" in decorations[p.name]:
                        with viz.subgraph() as s:
                            s.attr(rank='same', rankdir='TB')
                            s.node(str(id(p)), marking_label, fontsize='34', shape="ellipse", style="filled",
                                   fillcolor=fillcolor)
                            _name = f"label_{str(id(p))}"
                            s.node(_name, decorations[p.name]['label'], shape='plaintext', fontsize='14',
                                   fixedsize='true', width='0', height='0', margin='0')
                            s.edge(str(id(p)), _name, style='invis', weight='100', minlen='0')
                    else:
                        viz.node(str(id(p)), marking_label, fontsize="34", shape="ellipse", style="filled",
                                 fillcolor=fillcolor)
                else:
                    if p.name in decorations and "label" in decorations[p.name]:
                        with viz.subgraph() as s:
                            s.attr(rank='same', rankdir='TB')
                            s.node(str(id(p)), marking_label, fontsize='34', fixedsize='true', shape="circle",
                                   width='0.75', style="filled", fillcolor=fillcolor)
                            _name = f"label_{str(id(p))}"
                            s.node(_name, decorations[p.name]['label'], shape='plaintext', fontsize='14',
                                   fixedsize='true', width='0', height='0', margin='0')
                            s.edge(str(id(p)), _name, style='invis', weight='100', minlen='0')
                    else:
                        viz.node(str(id(p)), marking_label, fontsize="34", fixedsize='true', shape="circle",
                                 width='0.75', style="filled", fillcolor=fillcolor)
        elif p in final_marking:
            if p.name in decorations and "label" in decorations[p.name]:
                with viz.subgraph() as s:
                    s.attr(rank='same', rankdir='TB')
                    s.node(str(id(p)), "<&#9632;>", fontsize='34', shape='doublecircle', fixedsize='true', width='0.75',
                           style="filled", fillcolor=fillcolor)
                    _name = f"label_{str(id(p))}"
                    s.node(_name, decorations[p.name]['label'], shape='plaintext', fontsize='14', fixedsize='true',
                           width='0', height='0.1', margin='0')
                    s.edge(str(id(p)), _name, style='invis', weight='100', minlen='0')
            else:
                viz.node(str(id(p)), "<&#9632;>", fontsize="32", shape='doublecircle', fixedsize='true', width='0.75',
                         style="filled", fillcolor=fillcolor)
        else:
            if p.name in decorations and "label" in decorations[p.name]:
                with viz.subgraph() as s:
                    s.attr(rank='same', rankdir='TB')
                    s.node(str(id(p)), "", shape='circle', fixedsize='true', width='0.75', style="filled",
                           fillcolor=fillcolor)
                    _name = f"label_{str(id(p))}"
                    s.node(_name, decorations[p.name]['label'], shape='plaintext', fontsize='14', fixedsize='true',
                           width='0', height='0', margin='0')
                    s.edge(str(id(p)), _name, style='invis', weight='100', minlen='0')
            else:
                viz.node(str(id(p)), label, shape='circle', fixedsize='true', width='0.75', style="filled",
                         fillcolor=fillcolor)

    # add arcs, in order by their source and target objects names, to avoid undeterminism in the visualization
    arcs_sort_list = sorted(list(net.arcs), key=lambda x: (x.source.name, x.target.name))

    # check if there is an arc with weight different than 1.
    # in that case, all the arcs in the visualization should have the arc weight visible
    arc_weight_visible = False
    for arc in arcs_sort_list:
        if arc.weight != 1:
            arc_weight_visible = True
            break

    for a in arcs_sort_list:
        penwidth = decorations[a]["penwidth"] if a in decorations and "penwidth" in decorations[a] else None
        label = decorations[a]["label"] if a in decorations and "label" in decorations[a] else ""
        color = decorations[a]["color"] if a in decorations and "color" in decorations[a] else None

        if not label and arc_weight_visible:
            label = a.weight

        label = str(label)
        arrowhead = "normal"

        viz.edge(str(id(a.source)), str(id(a.target)), label=label,
                 penwidth=penwidth, color=color, fontsize=font_size, arrowhead=arrowhead, fontcolor=color)

    viz.attr(labelloc='b', labeljust='c')

    viz.format = image_format.replace("html", "plain-ext")

    return viz


def plot_and_save_net(m, im=None, fm=None, decorations: dict = None, output_file_path: str = './petri_net_img.png',
                      show: bool = True):
    if im is None:
        im = ["p_i"]
    if fm is None:
        # fm = ["p_f"]
        fm = []

    if isinstance(m, dict):
        net, im, fm = to_SPN(m, initial_marking=im, final_marking=fm)
    elif isinstance(m, tuple):
        assert len(m) == 3
        net, im, fm = m
    else:
        raise ArgumentError

    gviz = _graphviz_visualization(net=net, initial_marking=im, final_marking=fm, decorations=decorations)
    # gviz.render(format="png", dpi=300, cleanup=True) #
    if show:
        pn_visualizer.view(gviz)
    pn_visualizer.save(gviz, output_file_path)


def simple_plot_and_save_net(spn: StochasticPetriNet, init_marking: Marking, filename: str):
    plot_and_save_net(
        (spn, init_marking, []),
        decorations={
            _t.name: {'weight': f"{_t.weight:.5f}"}
            for _t in spn.transitions
        },
        output_file_path=filename,
        show=False
    )
