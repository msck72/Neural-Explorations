from graphviz import Digraph


def draw_graph(last_node):
    dot = Digraph(comment='Left to Right Tree', format='png')
    dot.attr(rankdir='LR')
    
    ptr = 0
    nodes = [last_node]
    visited = set()
    dot.node(name=str(id(last_node)), label = "{ data %.4f | grad %.4f }" % (last_node.data, last_node.grad), shape='record')

    while(ptr < len(nodes)):
        curr_node = nodes[ptr]

        # print(curr_node)
        if curr_node in visited or curr_node._formed_by is None:
            ptr += 1
            continue
        visited.add(curr_node)
        
        if curr_node._op is not None:
            dot.node(name=str(id(curr_node)) + curr_node._op, label=f'{curr_node._op}')
            dot.edge(str(id(curr_node)) + curr_node._op, str(id(curr_node)))

        for fb in curr_node._formed_by:
            dot.node(name=str(id(fb)), label = "{ data %.4f | grad %.4f }" % (fb.data, fb.grad), shape='record')
            nodes.append(fb)
            dot.edge(str(id(fb)), str(id(curr_node)) + curr_node._op)
        ptr += 1

    return dot
