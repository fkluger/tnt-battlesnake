#include "snake.hpp"
#include <cstdlib> 
#include <ctime> 
#include <algorithm>

Snake::Snake(coords head, int idx){
	m_health = m_maxhealth  = 100;
    srand((unsigned)time(0));
    m_head_direction = static_cast<direction>(rand()%4);
    m_body = std::vector<coords>{head};
    m_idx = idx;
	m_alive = true;
    m_max_length = 3;
}

void Snake::set_health(int health) {
	m_health = m_health * health / m_maxhealth;
	m_maxhealth = health;
	//std::min(m_health, health);
}

void Snake::move_head(int action){
    m_health--;
    if (is_dead()){
        return;
    }
    direction move_direction = get_direction(action);
    coords next_head = get_next_head(move_direction);

    m_body.insert(m_body.begin(), next_head);
    m_head_direction = move_direction;
}

void Snake::move_tail(bool ate_fruit){
    if (ate_fruit){
        m_max_length++;
        m_health = m_maxhealth;
    }
    if (m_body.size() > m_max_length){
        m_body.pop_back();
    }
}

bool Snake::is_dead(){
    return !m_alive;
}

int Snake::get_health() {
	return m_health;
}

coords Snake::get_head(){
    return m_body[0];
}

int Snake::get_length()
{
	return m_body.size();
}

void Snake::die(){
    m_health = 0;
	m_alive = false;
    m_body.clear();
}

void Snake::dead(){
    m_health = 0;
}

int Snake::get_idx(){
    return m_idx;
}

std::vector<coords> Snake::get_body(){
    return m_body;
}

direction Snake::get_direction(int action){
    if (m_head_direction == up){
        if (action == 0){
            return left;
        } else if (action == 1){
            return up;
        } else {
            return right;
        }
    } else if (m_head_direction == right){
        if (action == 0){
            return up;
        } else if (action == 1){
            return right;
        } else{
            return down;
        }
    } else if (m_head_direction == down){
        if(action == 0){
            return right;
        } else if (action == 1){
            return down;
        } else{
            return left;
        }
    } else{
        if (action == 0){
            return down;
        } else if (action == 1){
            return left;
        } else{
            return up;
        }
    }
}

coords Snake::get_next_head(direction dir){
    coords head = m_body[0];
    if (dir == up){
        return coords(head.x, head.y - 1);
    }else if (dir == right){
        return coords(head.x + 1, head.y);
    }else if (dir == down){
        return coords(head.x, head.y + 1);
    }else{
        return coords(head.x - 1, head.y);
    }
}
