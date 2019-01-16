#include "mcts_agent.hpp"
#include "mcts_parallel.h"

MCTSAgent::MCTSAgent(float time, int num_actions, int idx, int health, bool parallel, bool use_fruits) : m_idx(idx), m_health(health) {
	m_parallel = parallel;
	if (parallel) {
		m_mcts = new MCTSParallel(time, num_actions, use_fruits);
	} else {
		m_mcts = new MCTS(time, num_actions, use_fruits);
	}
}

MCTSAgent::~MCTSAgent()
{
	//delete m_mcts;
}

int MCTSAgent::act(State state){
	state.set_health(m_health);
    state.set_current_player(m_idx);
    return m_mcts->get_action(state);
}
