#include "uct_node.hpp"
#include <numeric>
#include <algorithm>
#include <math.h> 
#include <iostream>

UCTNode::UCTNode(State state, int active_player, int action, int num_actions, int winner, UCTNode* parent):
    m_state(state), m_active_player(active_player), m_action(action), m_num_actions(num_actions), 
	m_winner(winner), m_parent(parent){
        m_isleaf = true;
		m_visits = 0;
		m_value = 0;
        m_children = std::vector<UCTNode*>(m_num_actions, nullptr);
    }

UCTNode::UCTNode(State state, int active_player, int action, int num_actions, std::vector<UCTNode*> children):
	m_state(state), m_active_player(active_player), m_action(action), m_num_actions(num_actions)
{	
	m_isleaf = false;
	m_winner = -2;
	m_parent = nullptr;
	for (auto& child : children) {
		m_visits += child->get_visits();
		m_value += child->get_visits() - child->get_value();
		child->set_parent(this);
		m_children.push_back(child);
	}
}

UCTNode::UCTNode() {

}


UCTNode::~UCTNode(){
	for (std::vector< UCTNode* >::iterator it = m_children.begin(); it != m_children.end(); ++it)
	{
		delete (*it);
	}
	m_children.clear();
}

float UCTNode::get_visits(){
	return m_visits;
}


void UCTNode::set_visits(float visits){
   m_visits = visits;
}

float UCTNode::get_value(){
	return m_value;
}

void UCTNode::set_value(float value){
	m_value = value;
}

int UCTNode::get_winner()
{
	return m_winner;
}

UCTNode* UCTNode::get_child(int best_action){
    return m_children[best_action];
}

void UCTNode::set_child(int best_action, UCTNode* node){
    m_children[best_action] = node;
}

State UCTNode::get_state(){
    return m_state;
}

UCTNode* UCTNode::select(){
    UCTNode* current_node = this;
    while(!current_node->is_leaf()){
        std::vector<float> v = current_node->child_uct();
        auto result = std::max_element(v.begin(), v.end());
        int best_action = std::distance(v.begin(), result);
		if (!current_node->get_child(best_action)) {
			State state = current_node->get_state();
			int winner = state.move_snake(best_action);
			int active_player = state.get_current_player();
			current_node->set_child(best_action, new UCTNode(
				state, active_player, best_action, m_num_actions, winner, current_node
			));
		}
        current_node = current_node->get_child(best_action);
    }
    return current_node;
}

bool UCTNode::is_leaf(){
    return m_isleaf;
}

void UCTNode::expand(){
    m_isleaf = false;
}

UCTNode* UCTNode::get_parent(){
    return m_parent;
}

int UCTNode::get_active_player(){
    return m_active_player;
}

void UCTNode::backup(int winner, int fruits_eaten){
    UCTNode* current_node = this;
    do {
        current_node->set_visits(current_node->get_visits() + 1);
        if (winner == current_node->get_active_player()){
            current_node->set_value(current_node->get_value() + 3 + fruits_eaten);
		}
		else {
			current_node->set_value(current_node->get_value() - 3 + fruits_eaten);
		}
        current_node = current_node->get_parent();
	} while (current_node);
}

float UCTNode::uct(float child_value, float child_visits){
    return (
        child_value / (child_visits + 1) 
        + sqrt(2 * log(get_visits()+1) / (child_visits + 1))
    );
}

void UCTNode::set_parent(UCTNode * parent)
{
	m_parent = parent;
}

std::vector<float> UCTNode::child_uct(){
    std::vector<float> child_uct_vec;
    for (auto& child : m_children){
		if (child) {
			child_uct_vec.push_back(uct(child->get_value(), child->get_visits()));
		}
		else {
			child_uct_vec.push_back(1000);
		}
    }
    return child_uct_vec;
}

std::vector<float> UCTNode::res(){
    std::vector<float> res_vec;
	for (auto& child : m_children) {
		if (child) {
			res_vec.push_back(child->get_value() / (child->get_visits()));
		}
    }
    return res_vec;
}
  