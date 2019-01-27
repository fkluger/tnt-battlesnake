#include "uct_node.hpp"
#include <numeric>
#include <algorithm>
#include <math.h> 
#include <sstream>
#include <iostream>

UCTNode::UCTNode(State state, int active_player, int action, int num_actions, int winner, UCTNode* parent):
    m_state(state), m_active_player(active_player), m_action(action), m_num_actions(num_actions), 
	m_winner(winner), m_parent(parent){
        m_isleaf = true;
		m_visits = 0;
		m_value = 0;
        m_children = std::vector<UCTNode*>(m_num_actions, nullptr);
    }

UCTNode::UCTNode(State state, int active_player, int action, int num_actions, std::vector<UCTNode*>& children):
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

void fill_spaces(int dim, std::ofstream& file){
	for (int j = 0; j < dim; ++j){
		file << "\t";
	}
}

std::vector<std::string> UCTNode::to_json() {
	std::vector<std::string> node_json;
	const void * address = static_cast<const void*>(this);
	std::stringstream ss;
	ss << address;  
	std::string name = ss.str(); 
	node_json.push_back("\tID: " + name);
	std::string root = ((m_parent != nullptr)? "1" : "0");
	node_json.push_back("\tRoot: " + root);
	node_json.push_back("\tVisits: " + std::to_string(m_visits));
	node_json.push_back("\tValue: " + std::to_string(m_value));
	node_json.push_back("\tChildren: [\n");
	return node_json;
}

void UCTNode::save_tree(int dim, std::ofstream& uct_file){
	std::vector<std::string> node_json = to_json();
	for (int i = 0; i < node_json.size(); ++i){
		fill_spaces(dim, uct_file);
		uct_file << node_json[i];
		uct_file << std::endl;
	}
	for (int i = 0; i < m_children.size(); ++i)
	{
		if(m_children[i] != nullptr){
			fill_spaces(dim + 1, uct_file);
			uct_file << "{" << std::endl;
			m_children[i]->save_tree(dim + 1, uct_file);
			
			fill_spaces(dim + 1, uct_file);
			uct_file << "}";
			if(i < m_children.size() - 1 && m_children[i+1] != nullptr){
				uct_file << ",";
			}
			uct_file << std::endl;
		}
	}
	fill_spaces(dim + 1, uct_file);
	uct_file << "]" << std::endl;
	
	 
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
            current_node->set_value(current_node->get_value() + 1);// + fruits_eaten);
		}
		else {
			current_node->set_value(current_node->get_value()); // + fruits_eaten);
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
  