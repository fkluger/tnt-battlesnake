#pragma once
#include "state.hpp"
#include "mcts.hpp"
#include "Agent.h"

class MCTSAgent : Agent{
    public:
        MCTSAgent(float time, int num_actions, int idx, int health = 100, bool parallel = false, bool use_fruits = true);
		~MCTSAgent();
		int act(State state) override;
    private:
        float m_time;
        int m_num_actions;
        int m_idx;
		int m_health;
        MCTS* m_mcts;
		bool m_parallel;
};
