#!/usr/bin/env python3
"""Render the published IEEE30 topology in the project's numbered-bus style.

Connectivity: IEEE CDF/MATPOWER case_ieee30 plus the six terminal transformers
shown in the Demetriou/KIOS modified network. This is a topology illustration,
not a PowerWorld import or verification of the pending dynamic backend.
"""
from pathlib import Path
import argparse
import json
import re
import hashlib

ROOT = Path(__file__).resolve().parents[1]


def reference_graph(config):
    import networkx as nx
    source = ROOT / 'tools/reference_data/case_ieee30.m'
    text = source.read_text()
    rows = re.search(r'mpc\.branch\s*=\s*\[(.*?)\];', text, re.S).group(1)
    edges = []
    for row in rows.splitlines():
        row = row.split('%')[0].strip().rstrip(';')
        if row:
            a, b = row.split()[:2]
            edges.append((int(a), int(b)))
    mapping = {int(k): int(v) for k, v in config['network']['original_to_terminal_bus'].items()}
    if len(edges) != 41 or set(mapping) != {1, 2, 5, 8, 11, 13}:
        raise ValueError('Unexpected reference network')
    graph = nx.Graph()
    graph.add_nodes_from(range(1, 37))
    graph.add_edges_from(edges)
    graph.add_edges_from(mapping.items())
    machines = config['dynamic_machines']
    if (not nx.is_connected(graph) or graph.number_of_edges() != 47
            or {m['terminal_bus'] for m in machines} != set(range(31, 37))):
        raise ValueError('Topology and machine data disagree')
    return graph, mapping, source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/ieee30_mocu.yaml'))
    parser.add_argument('--output', type=Path, default=Path('documents/images/ieee30_diagram.png'))
    args = parser.parse_args()
    import yaml
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import networkx as nx
    config = yaml.safe_load(args.config.read_text())
    if config.get('system', {}).get('variant') != 'demetriou2017_modified_dynamic':
        parser.error('This renderer requires the Demetriou IEEE30 reference configuration')
    graph, mapping, source = reference_graph(config)
    # Fixed reproducible schematic coordinates; intersections without numbered
    # circles are crossings, not electrical junctions.
    pos = {
        1:(0,8), 2:(3,8), 3:(0,6.5), 4:(2,6.5), 5:(6,8),
        6:(4.5,6.5), 7:(6.2,7.2), 8:(8,6.5), 9:(4.5,4.8),
        10:(6,3.5), 11:(3.1,4.8), 12:(1,4.8), 13:(-0.4,4.8),
        14:(0,3.2), 15:(1.5,3.2), 16:(2.8,3.5), 17:(4,2.3),
        18:(1.5,1.6), 19:(3,1.6), 20:(4.8,1.6), 21:(7,3.3),
        22:(7.6,1.1), 23:(3,0), 24:(5.8,0), 25:(7.7,-1),
        26:(9,-1.5), 27:(9.5,1.2), 28:(9.5,5), 29:(11,1.9),
        30:(11,-0.2), 31:(-1,9.1), 32:(3,9.5), 33:(6.4,9.5),
        34:(8,8.2), 35:(3.1,5.85), 36:(-1.8,4.8),
    }
    # An unrelated bus must not visually split a direct branch.
    for node,(x,y) in pos.items():
        for a,b in graph.edges:
            if node in (a,b):
                continue
            ax0,ay0=pos[a];bx,by=pos[b]
            dx,dy=bx-ax0,by-ay0
            t=max(0,min(1,((x-ax0)*dx+(y-ay0)*dy)/(dx*dx+dy*dy)))
            distance=((x-ax0-t*dx)**2+(y-ay0-t*dy)**2)**.5
            if distance < .28:
                raise ValueError(f'Bus {node} obscures unrelated branch {a}-{b}')
    fig, ax = plt.subplots(figsize=(14, 12))
    fig.patch.set_facecolor('white')
    base_transformers = {frozenset(e) for e in [(6,9),(6,10),(4,12),(27,28)]}
    added = {frozenset(e) for e in mapping.items()}
    for a,b in graph.edges:
        transformer = frozenset((a,b)) in base_transformers | added
        ax.plot([pos[a][0],pos[b][0]], [pos[a][1],pos[b][1]],
                color='black', linewidth=2.0, linestyle='--' if transformer else '-', zorder=1)
    nodes=list(graph.nodes)
    colors=['#f9df45' if n==31 else '#4caf50' if n>=32 else '#4a90d9' for n in nodes]
    nx.draw_networkx_nodes(graph,pos,nodelist=nodes,node_color=colors,
                           edgecolors='black',linewidths=2,node_size=760,ax=ax)
    nx.draw_networkx_labels(graph,pos,font_size=14,font_weight='bold',ax=ax)
    for terminal,role in [(31,'G1'),(32,'G2'),(33,'SC5'),(34,'SC8'),(35,'SC11'),(36,'SC13')]:
        x,y=pos[terminal]
        ax.annotate(role,(x,y),xytext=(0,24),textcoords='offset points',
                    ha='center',fontsize=10,fontweight='bold')
    handles=[Line2D([],[],marker='o',color='none',markerfacecolor=c,
                    markeredgecolor='black',markersize=12,label=l)
             for c,l in [('#f9df45','Slack generator terminal'),
                         ('#4caf50','Generator / condenser terminal'),
                         ('#4a90d9','Algebraic network bus')]]
    handles += [Line2D([],[],color='black',linewidth=2,label='Transmission line'),
                Line2D([],[],color='black',linewidth=2,linestyle='--',label='Transformer')]
    fig.suptitle('IEEE 30-bus modified network (Demetriou et al., 2017)',fontsize=20,fontweight='bold',y=.975)
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.94),ncol=3,fontsize=11,frameon=True)
    ax.set_xlim(-2.5,11.8);ax.set_ylim(-2.2,10.4);ax.set_aspect('equal');ax.set_axis_off()
    fig.text(.5,.035,'36 electrical buses | 6 dynamic machines: 2 generators + 4 synchronous condensers',
             ha='center',fontsize=12)
    fig.text(.5,.016,'G: governed generator; SC: synchronous condenser. Crossings without circles are not junctions.',
             ha='center',fontsize=10)
    fig.subplots_adjust(top=.85,bottom=.075,left=.03,right=.97)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=180,facecolor='white')
    plt.close(fig)
    metadata={
        'config':str(args.config),'paper_doi':'10.1109/JSYST.2015.2444893',
        'reference_page':'https://www.kios.ucy.ac.cy/testsystems/index.php/ieee-30-bus-modified-test-system/',
        'branch_source':str(source.relative_to(ROOT)),
        'branch_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'construction':'41 original IEEE CDF branches plus 6 published machine-terminal transformers',
        'nodes':nodes,'edges':sorted([sorted(e) for e in graph.edges]),
        'machine_terminals':sorted(mapping.values()),'electrical_bus_count':36,
        'transmission_lines':37,'transformers':10,
        'scope':'Connectivity illustration; not a validated full dynamic simulation',
    }
    args.output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(args.output)


if __name__=='__main__':
    main()
