    #include<iostream>
    #include<vector>
    #include<random>
    
    using namespace std;
    
    // Define the struct for FSC node
    struct FscNode{
        // particles of the node's state
        vector<double>_state_particles;
        
        // dictionary for Q-action
        Dict<Any, double>_Q_action;
        
        // dictionary for reward action
        Dict<Any, double>_R_action;
        
        // dictionary for value of action-observation
        Dict<Any, Dict<Int64, double>>_V_a_o_n;
        
        // value of the node
        double _V_node;
    };
    
    // Define the struct for FSC
    struct FSC{
        // vector of pairs for eta
        vector<Dict<Pair<Any, Int64>, Int64>>_eta;
        
        // vector of nodes
        vector<FscNode>_nodes;
        
        // action space
        Dict<Int64, Int64>_action_space;
    };
    
    // Function to initialize FSC node
    function InitFscNode(action_space, obs_space){
        // initialize particles, Q-action, reward action, value of action-observation, and value of the node
        vector<double> init_particles;
        Dict<Any, double> init_Q_action = Dict<Any, double>();
        Dict<Any, double> init_R_action = Dict<Any, double>();
        Dict<Any, Dict<Int64, double>> init_V_a_o_n = Dict<Any, Dict<Int64, double>>();
        for (int a =0; a< action_space.size(); a++){
            init_Q_action[a]=0.0;
            init_R_action[a]=0.0;
            init_V_a_o_n[a]= Dict<Int64, double>();
            for (int o =0; o< obs_space.size(); o++){
                init_V_a_o_n[a][o]=  Dict<Int64, double>();
            }
        }
        
        // return the FSC node with the initial values
        return FscNode(init_particles,
                        init_Q_action,
                        init_R_action,
                        init_V_a_o_n,
                        init_V_node);
    }
    
    // Function to create node
    function CreatNode(b, action_space){
        // initialize FSC node with the given action space
        FscNode node = InitFscNode(action_space);
        node._state_particles = b;
        return node;
    }
    
    // Function to initialize FSC with given parameters
    function InitFSC(max_accept_belief_gap::Float64, max_node_size::Int64, action_space){
        // initialize eta vector with given max node size
        vector<Dict<Pair<Any, Int64>, Int64>> init_eta(max_node_size);
        for (int i =1; i<= max_node_size; i++){
            init_eta[i-1]= Dict<Pair<Any, Int64>, Int64>();
        }
        
        // return the FSC with the initial eta and nodes
        return FSC(init_eta,
                    init_nodes,
                    action_space);
    }
    
    // Function to get the best action
    function GetBestAction(n::FscNode){
        // get the maximum Q-value
        double Q_max = typemin<double>();
        int best_a = rand(n._Q_action.keys())
        for (int a =0; a< n._Q_action.size(); a++){
            if (n._Q_action[a]> Q_max&& n._visits_action[a]!=0){
                Q_max = n._Q_action[a];
                best_a = a;
            }
        }
    
        return best_a;
    }