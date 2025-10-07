
import argparse
import random
import numpy as np

from supernode_net import make_random_network, suggest_radius

def log_path(path):
    if not path:
        print("   (no path)" )
        return
    print(f"   hops: {len(path)-1}") 
    print(f"   path head: {path[:8]}{'...' if len(path)>8 else ''}")
    print("   full path:") 
    print("   " + " -> ".join(map(str, path)))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('-n', type=int, default=150, help='number of nodes')
    p.add_argument('-L', type=float, default=1.0, help='field side length')
    p.add_argument('--radius', type=float, default=None, help='connection radius (default: suggested)')
    p.add_argument('--seed', type=int, default=11, help='random seed')
    p.add_argument('--super-frac', type=float, default=0.1, help='fraction of supernodes')
    p.add_argument('--super-mode', type=str, default='degree', choices=['degree','battery'], help='supernode selection metric')
    p.add_argument('--method', type=str, default='all', choices=['shortest','greedy','supernode','all'], help='routing method to run')
    p.add_argument('--show-ids', action='store_true', help='draw node IDs (hidden if >200 nodes)')
    p.add_argument('--save-dir', type=str, default='/mnt/data', help='directory to save figures')
    args = p.parse_args()

    print("--- Building random network ---")
    print(f"Nodes: {args.n}, Field: {args.L}x{args.L}, Seed: {args.seed}")

    net = make_random_network(n=args.n, L=args.L, radius=args.radius, seed=args.seed,
                              super_frac=args.super_frac, super_mode=args.super_mode)
    print(f"Suggested connection radius: {suggest_radius(args.n, args.L):.3f}")
    degs = [len(net.nodes[i].neighbors) for i in net.nodes]
    print(f"Avg degree: {np.mean(degs):.2f}, Min degree: {np.min(degs)}, Max degree: {np.max(degs)}")
    super_ids = [i for i, nd in net.nodes.items() if nd.is_supernode]
    print(f"Supernodes (count={len(super_ids)}): {sorted(super_ids)[:12]}{'...' if len(super_ids)>12 else ''}")

    # plot topology
    topo_path = f"{args.save_dir}/topology.png"
    net.plot_topology(title="Random Network with Supernodes", show_ids=args.show_ids, save_path=topo_path)
    print(f"Saved topology: sandbox:{topo_path}")

    # choose random connected (s,t)
    rng = random.Random(args.seed)
    ids = list(net.nodes.keys())
    for _ in range(20):
        s, t = rng.sample(ids, 2)
        if net.bfs_shortest_path(s, t):
            break
    else:
        print("Warning: could not find connected (s,t)")
        return

    if args.method in ('shortest','all'):
        sp = net.bfs_shortest_path(s, t)
        print(f"\n[Shortest-Path] s={s} -> t={t}")
        log_path(sp)
        out = f"{args.save_dir}/route_shortest.png"
        net.plot_route(sp, title=f"Shortest-Path Route ({len(sp)-1 if sp else 'NA'} hops)", save_path=out)
        print(f"Saved: sandbox:{out}")

    if args.method in ('greedy','all'):
        gr = net.greedy_route(s, t)
        print(f"\n[Greedy] s={s} -> t={t}")
        log_path(gr)
        out = f"{args.save_dir}/route_greedy.png"
        net.plot_route(gr, title=f"Greedy Route ({len(gr)-1 if gr else 'NA'} hops)", save_path=out)
        print(f"Saved: sandbox:{out}")

    if args.method in ('supernode','all'):
        sr = net.supernode_route(s, t)
        print(f"\n[Supernode] s={s} -> t={t}")
        log_path(sr)
        out = f"{args.save_dir}/route_supernode.png"
        net.plot_route(sr, title=f"Supernode Route ({len(sr)-1 if sr else 'NA'} hops)", save_path=out)
        print(f"Saved: sandbox:{out}")

    print("\nDone.")

if __name__ == '__main__':
    main()
