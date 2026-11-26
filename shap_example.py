import itertools
import math
from pymapf import pymapf

# ---------------------------------------------------------
# 1. Define 2D Obstacle Grid & Agents
#    0 = free cell, 1 = static obstacle
#    Central row (row 3) has a 1-cell choke point at (3, 3)
# ---------------------------------------------------------
GRID = pymapf.GridMap([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 1, 0],
    [0, 1, 1, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0],  # Horizontal corridor + vertical choke point
    [0, 1, 1, 0, 1, 1, 0],
    [0, 1, 1, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0],
])

# Agents a0 & a1 cross paths through (3, 3); a2 operates in an isolated corner
AGENTS = [
    pymapf.Agent("a0", start=(3, 1), goal=(3, 5)),
    pymapf.Agent("a1", start=(3, 5), goal=(3, 1)),
    pymapf.Agent("a2", start=(6, 4), goal=(6, 6)),
]


# ---------------------------------------------------------
# 2. Solver Wrapper & Cost Metrics
# ---------------------------------------------------------
def solve_instance(
    grid: pymapf.GridMap,
    agents: list[pymapf.Agent],
    solver_name: str = "cbs",
):
    """Solves a one-shot MAPF instance and returns (solution, sum_of_costs, makespan)."""
    if not agents:
        return None, 0, 0

    problem = pymapf.MAPFProblem(grid, agents)
    solution = pymapf.solve(problem, solver_name)
    return solution, solution.sum_of_costs, solution.makespan


# ---------------------------------------------------------
# 3. Coalition Value Function v(S) & Exact Shapley Attribution
# ---------------------------------------------------------
def compute_shapley_blame(
    grid: pymapf.GridMap,
    agents: list[pymapf.Agent],
    solver_name: str = "cbs",
) -> tuple[dict[str, float], float]:
    """
    Computes exact Shapley attribution for multi-agent coordination delay:
        v(S) = SumOfCosts(Sim(S)) - sum_{i in S} SoloShortestPath(i)
    """
    n_agents = len(agents)
    indices = list(range(n_agents))

    # Baseline unobstructed shortest-path cost for each individual agent
    solo_costs = {
        i: solve_instance(grid, [agents[i]], solver_name)[1]
        for i in indices
    }

    # Cache coalition values v(S) across all 2^N subsets
    v_cache: dict[tuple[int, ...], float] = {(): 0.0}
    for r in range(1, n_agents + 1):
        for subset in itertools.combinations(indices, r):
            sub_agents = [agents[i] for i in subset]
            _, coalition_soc, _ = solve_instance(grid, sub_agents, solver_name)
            ideal_soc = sum(solo_costs[i] for i in subset)
            v_cache[subset] = float(coalition_soc - ideal_soc)

    # Exact Shapley formula
    shapley_scores = {agents[i].name: 0.0 for i in indices}
    n_fact = math.factorial(n_agents)

    for i in indices:
        others = [j for j in indices if j != i]
        for r in range(len(others) + 1):
            weight = (math.factorial(r) * math.factorial(n_agents - r - 1)) / n_fact
            for subset in itertools.combinations(others, r):
                s_without = tuple(sorted(subset))
                s_with = tuple(sorted(subset + (i,)))
                marginal_delay = v_cache[s_with] - v_cache[s_without]
                shapley_scores[agents[i].name] += weight * marginal_delay

    total_system_delay = v_cache[tuple(indices)]
    return shapley_scores, total_system_delay


# ---------------------------------------------------------
# 4. Run Solver Comparison & Blame Attribution
# ---------------------------------------------------------
if __name__ == "__main__":
    solvers = ["cbs", "wcbs", "prioritized", "pibt", "lacam", "lns"]

    print("=== Multi-Solver Comparison ===")
    for solver in solvers:
        try:
            sol, soc, ms = solve_instance(GRID, AGENTS, solver_name=solver)
            blame, total_delay = compute_shapley_blame(GRID, AGENTS, solver_name=solver)
            print(f"\nSolver: {solver.upper()} (vali d={sol.is_valid()})")
            print(f"  Sum-of-Costs: {soc} | Makespan: {ms} | Total Coordination Delay: {total_delay}")
            for agent_name, phi in blame.items():
                print(f"  Agent {agent_name} Blame (phi_{agent_name}): {phi:.2f} steps")
        except Exception as e:
            print(f"\nSolver {solver} skipped: {e}")

    # Inspect CBS paths and optionally save a plot if pymapf[viz] is installed
    cbs_problem = pymapf.MAPFProblem(GRID, AGENTS)
    cbs_sol = pymapf.solve(cbs_problem, "cbs")
    print("\n=== CBS Trajectories ===")
    for name, path in cbs_sol.paths.items():
        print(f"  {name}: {path}")

    try:
        from pymapf import viz
        fig = viz.plot_solution(cbs_sol, cbs_problem)
        viz.save(fig, "cbs_plan.png")
        print("Saved plot to cbs_plan.png")
    except ImportError:
        pass