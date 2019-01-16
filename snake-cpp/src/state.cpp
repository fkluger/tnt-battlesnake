#include "state.hpp"
#include <cstdlib> 
#include <ctime> 
#include <algorithm>


State::State(int width, int height, int num_snakes, int num_fruits):
    m_width(width), m_height(height), m_num_snakes(num_snakes), m_num_fruits(num_fruits) {
        place_fruits_or_snakes(num_fruits, true);
        place_fruits_or_snakes(num_snakes, false);
        update_snakes_alive();
        m_current_snake_idx = 0;
        m_alive_idx = 0;
}

State::State(const State & old_state){
	m_alive_idx = old_state.m_alive_idx;
	m_current_snake_idx = old_state.m_current_snake_idx;
	m_fruits = old_state.m_fruits;
	m_height = old_state.m_height;
	m_num_fruits = old_state.m_num_fruits;
	m_num_snakes = old_state.m_num_snakes;
	m_snakes = old_state.m_snakes;
	m_width = old_state.m_width;

	update_snakes_alive();
}


void State::set_current_player(int idx){
    std::rotate(m_snakes.begin(), m_snakes.begin() + idx, m_snakes.end());
    update_snakes_alive();
}

int** State::get_array(){
    int** array = 0;
    array = new int*[m_height];
    for (int y = 0; y < m_height; ++y){
        array[y] = new int[m_width];
        for (int x = 0; x < m_width; ++x){
            if (x == 0 || x == m_width - 1 || y == 0 || y == m_height - 1){
                array[y][x] = 40;
            } else {
                array[y][x] = 0;
            }
        }
    }

    for (int j = 0; j < m_snakes_alive.size(); ++j){
        std::vector<coords> body = m_snakes_alive[j]->get_body();
        for (int i = 0; i < body.size(); ++i){
            array[body[i].y][body[i].x] = 50 * (m_snakes_alive[j]->get_idx() + 1);
        }
    }

    for (int j = 0; j < m_fruits.size(); ++j){
        array[m_fruits[j].y][m_fruits[j].x] = 11;
    }

    return array;
}

void State::set_health(int health)
{
	for (auto& snake : m_snakes) {
		snake.set_health(health);
	}
}

void State::update_snakes_alive(){
    m_snakes_alive.clear();
    for(auto& snake: m_snakes) {
        if(!snake.is_dead()){
            m_snakes_alive.push_back(&snake);
        }
    }
}

void State::place_fruits_or_snakes(int fields, bool is_fruit){
    int padding = 2;
    if (is_fruit){
        padding = 1;
    }
    srand((unsigned)time(0)); 
    for(int idx = 0; idx < fields; ++idx){
        coords field = coords(-1,-1);
        while (!is_available(field)){
            field = coords(
                    (rand()%(m_width-2*padding))+padding, 
                    (rand()%(m_height-2*padding))+padding
                    ); 
        }
        if (is_fruit){
            m_fruits.push_back(field);
        } else {
            m_snakes.push_back(Snake(field, idx));
        }
    }
}

bool State::is_available(coords field){
    if(field.x == -1 && field.y == -1){
        return false;
    }

    bool available = true;
    for (auto const& fruit_field: m_fruits){
        if(field == fruit_field){
            available = false;
        }
    }
    for (auto& snake: m_snakes){
        for(auto const& snake_field: snake.get_body()){
            if (field == snake_field){
                available = false;
            }
        }
    }
    return available;
}

int State::move_snake(int action){
    if (m_snakes_alive.size() == 0){
        return -1;
    }
    Snake* s = m_snakes_alive[m_alive_idx];
    m_current_snake_idx = s->get_idx();
    m_alive_idx++;
    s->move_head(action);
    if (m_alive_idx >= m_snakes_alive.size()){
        collision_check();

        int fruits_eaten = 0;
        for (auto& snake: m_snakes_alive){
            bool ate_fruite = ate_fruit(*snake);

            if (snake->get_health() <= 0){
                snake->die();
            } else{
                snake->move_tail(ate_fruite);
            }

            if (ate_fruite){
                fruits_eaten++;
            }
        }

        place_fruits_or_snakes(fruits_eaten, true);

        update_snakes_alive();
        m_alive_idx = 0;
        if (m_snakes_alive.size() == 0){
            return -1;
        } else if (m_snakes_alive.size() == 1){
            return m_snakes_alive[0]->get_idx();
        }
    }
    return -2;
}

Snake State::get_snake(int idx)
{
	return m_snakes[idx];
}

std::vector<coords> State::get_fruits()
{
	return m_fruits;
}

int State::get_current_player(){
    return m_current_snake_idx;
}


int State::move_snakes(std::vector<int> actions){
    for (int snake_idx = 0; snake_idx < m_snakes_alive.size(); ++snake_idx){
        int winner = move_snake(actions[snake_idx]);
        if (winner != -2){
            return winner;
        }
    }
    if (m_snakes_alive.size() == 0){
            return -1;
    }
    return 0;
}

bool State::ate_fruit(Snake snake){
    bool ate_fruit = false;
    coords snake_head = snake.get_head();
    std::vector<coords> new_fruits;
    for (auto& fruit: m_fruits){
        if (snake_head == fruit){
            ate_fruit = true;
        }else {
            new_fruits.push_back(fruit);
        }
    }
    m_fruits = new_fruits;
    return ate_fruit;
}

void State::collision_check(){
    for (auto& snake: m_snakes_alive){
        coords snake_head = snake->get_head();
        if (
            snake_head.x <= 0
            || snake_head.y <= 0
            || snake_head.x >= (m_width - 1)
            || snake_head.y >= (m_height - 1)
        ){
            snake->dead();
        }
        for (auto& other_snake: m_snakes_alive){
            std::vector<coords> s_body = other_snake->get_body();
            for (int s_body_idx = 0; s_body_idx < s_body.size(); ++s_body_idx){
                if (snake_head == s_body[s_body_idx]){
                    if (s_body_idx != 0){
                        snake->dead();
                    } else{
                        if (snake != other_snake && snake->get_body().size() <= s_body.size()){
                            snake->dead();
                        }
                    }
                }
            }
        }
    }
    return;
}