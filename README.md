kuka multi-object grasp task based on DDPG
====  
## Environment  

https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/baselines/enjoy_kuka_diverse_object_grasping.py

环境部分都是从这个`benchmark`摘出来的.
对环境做了修改:
* 删除了原始环境在对动作空间的伸缩. 原来动作空间(-1,1)*4 . 又在在kuka层进行了伸缩和剪切.

* 增加合法活动区域. 超出活动区域将会终止episode,且给一个负反馈.

* 修改抓取动作的触发高度

* 从模拟器提取 full-state 信息. ( 物体位姿 , 爪子位姿 ) 作为critic 的state输入. 促进训练.

该环境主要毛病是:
* 暂时只使用单核. 效率很低.

* 算法还没有调试到最好.(组合方法,超参数)

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