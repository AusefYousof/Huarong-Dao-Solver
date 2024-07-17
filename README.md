Description: Solve the Hua Rong Dao chinese sliding tile puzzle using DFS and A* (with pruning)
Hua Rong Dao rules: http://chinesepuzzles.org/huarong-pass-sliding-block-puzzle/Links to an external site.

Empty squares are represented with .  
1x1 Pieces are represented with a 2  
2x1 pieces are represented with a <>  
1x2 pieces are represented with ^  
                                v  
The goal 2x2 piece is represented with  11  
                                        11  
  
Objective is to move the 2x2 piece into the bottom middle slot.  
  
  
Cmd: python hrd.py --inputfile [inputfile name] --outputfile [outputfile name] --algo [algo name (dfs or astar)]