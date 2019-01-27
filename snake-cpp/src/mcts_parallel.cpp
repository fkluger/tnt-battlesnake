#include "mcts_parallel.h"
#include <chrono>
#include <thread>
#include <iostream>
#include <algorithm>
#include <future> 

MCTSParallel::MCTSParallel(float simulation_time, int num_actions, bool use_fruits)
{ 
	m_simulation_time = simulation_time;
	m_num_actions = num_actions;
	m_use_fruits = use_fruits;
}

MCTSParallel::MCTSParallel()
{
}

int MCTSParallel::get_action(State state)
{
	delete m_root_node;
	srand((unsigned)time(0));
	std::vector<UCTNode*> action_nodes(m_num_actions, nullptr);
	std::vector<std::thread> action_threads;

	for (int i = 0; i < m_num_actions; ++i) {
		State state_tmp = state;
		state_tmp.move_snake(i);
		action_nodes[i] = new UCTNode(
			state_tmp, state_tmp.get_current_player(), i, m_num_actions, -2, nullptr
		);
		action_threads.push_back(std::thread(&MCTSParallel::calc_tree, this, action_nodes[i]));
	}
	for (int j = 0; j < m_num_actions; ++j) {
		action_threads[j].join();
	}

	m_root_node = new UCTNode(
		state, 
		state.get_current_player(), 
		0, 
		m_num_actions, 
		action_nodes
	);

	std::vector<float> v = m_root_node->res();
	auto result = std::max_element(v.begin(), v.end());
	int best_action = std::distance(v.begin(), result);

	std::cout << m_root_node->get_visits() << std::endl;
	
	return best_action;

}

void MCTSParallel::calc_tree(UCTNode* root_node) {
	auto start_time = std::chrono::steady_clock::now();
	auto end_time = std::chrono::steady_clock::now();
	while (std::chrono::duration_cast<std::chrono::milliseconds>
		(end_time - start_time).count() < m_simulation_time * 1000 - 30) {
		UCTNode* leaf_node = root_node->select();
		int winner = leaf_node->get_winner();
		if (winner != -2) {
			leaf_node->backup(winner, 0);
			end_time = std::chrono::steady_clock::now();
			continue;
		}
		else {
			State discover_state = leaf_node->get_state();
			int snake_length = discover_state.get_snake(discover_state.get_current_player()).get_length();
			while (true) {
				winner = discover_state.move_snake(rand() % m_num_actions);
				if (winner != -2) {
					break;
				}
			}
			int snake_length_dif = snake_length - discover_state.get_snake(discover_state.get_current_player()).get_length();
			leaf_node->expand();
			leaf_node->backup(winner, (m_use_fruits)? snake_length_dif : 0);
		}
		end_time = std::chrono::steady_clock::now();
	}
}

