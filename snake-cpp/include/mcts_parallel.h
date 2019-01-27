#pragma once
#include "state.hpp"
#include "uct_node.hpp"
#include "mcts.hpp"

class MCTSParallel : public MCTS {
public:
	MCTSParallel(float simulation_time, int num_actions, bool use_fruits);
	MCTSParallel();
	int get_action(State state) override;
private:
	void calc_tree(UCTNode* root_node);
};