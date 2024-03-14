Goals 
- MC-JESP for solving MA-CTPs
    - [ ] Improve MC-JESP by using MCVI in each MC-JESP's iteration for solving the best-response POMDP
    - [ ] Further improvements considering (MA-)CTP problems' features about deterministic dynamics


Current MCVI C++ Implementation Progress
- Basic Interfaces
    - [x] POMDP Interface
    - [x] Simulator Interface
    - [ ] Finite State Controller 
    - [ ] Belief Expansion Tree

- Main Components
    - [ ] Upper Bound Evaluation
        - [x] Q-learning implementation
    - [ ] Lower Bound Evaluation (Simulation with the FSC built)
    - [ ] Belief Expand Method
    - [ ] Monte-Carlo BackUp
        - [x] Basic implementation
        - [ ] Testing
    - Testing with POMDP and CTP benchmarks
        - [ ] Tiger
        - [ ] Rock Sample
        - [ ] Alex's CTP problems

Current MCVI Julia Implementation Progress 
- Basic Interfaces
    - [x] POMDP Interface
    - [x] Simulator Interface
    - [x] Finite State Controller 
    - [ ] Belief Expansion Tree

- Main Components
    - [x] Upper Bound Evaluation
        - [x] Q-learning implementation
    - [x] Lower Bound Evaluation (Simulation with the FSC built)
    - [ ] Belief Expand Method
    - [x] Monte-Carlo BackUp
        - [x] Basic implementation
        - [x] Testing

- Testing with POMDP and CTP benchmarks
    - [ ] Tiger
    - [ ] Rock Sample
    - [ ] Alex's CTP problems