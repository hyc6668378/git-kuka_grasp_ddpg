kuka multi-object grasp task based on DDPG
====  
## Environment  

https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/baselines/enjoy_kuka_diverse_object_grasping.py

环境部分都是从这个`benchmark`摘出来的.
对环境做了修改:

* 增加合法活动区域. 超出活动区域将会终止episode,且给一个负反馈.

* 修改reward function.

* 修改抓取动作的触发高度

* 将低维full-state信息(物体位姿 , 爪子位姿)作为critic的state.

框架扩充：
* 支持mpi. (ok, 我承认，其实是照着baselines写的...)

(mpi对于交互开销不大的环境，并不会加快训练，反而会因为网络的同步更新，拖慢训练速度。)

## 算法选择
算法部分可以选择`标准DDPG` 和如下`improvements`之一
- [x] `Prioritied Memory Replay `
- [x] `n-step-ruturn`
- [x] `DDPGfD`
- [x] `TD3`
- [ ] `her`
- [ ] `curiosity`
......

## run
`./run_bash.sh`
