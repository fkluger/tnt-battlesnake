#pragma once
#include "state.hpp"
class Agent
{
public:
	Agent();
	~Agent();
	virtual int act(State state) { return 1; };
};

