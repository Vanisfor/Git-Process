
*1*. Git 四个区域: 远程，本地仓库，暂存区，工作区

![image](https://github.com/Vanisfor/test/blob/main/1728940033317.png)

```
一次提交多个文件

git add *.ipynb #同后缀的文件

git add file1.ipynb file2.ipynb file3.txt #分开指定

git add -u #只提交已被修改的文件
```

1）push一个新的本地仓库
```
[echo "# Machine-Learning" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main #如果你的分支名叫master 这里需要改成master 视情况而定
git remote add origin https://github.com/Vanisfor/Machine-Learning.git
git push -u origin main #无法push的话用下一行的

git push --force origin main  #-f 表示force 强制执行
```
git 初始化 （注意：global是全局 不加global就是当前）

2）push一个已经存在的仓库
```
git remote add origin https://github.com/Vanisfor/Machine-Learning.git
git branch -M main
git push -u origin main
```
3）git直接拉取代码
```
git clone https://github.com/...
```


1）配置使用者
```
 git config --global user.name "..."
```
2）配置使用者邮箱
```
git config --global user.email "xxx@qq/126/163.com"
```
3） 高亮
```
git config --global color.ui true
```

4）
```
git checkout #跳分支
```
5）

```
git status #查看文件状态
```

6）branch: 从main复制一个分支下来，方便团队协作


如果已经提交完，那么再次提交将被视为无修改文件，无法提交到暂存区

7）项目
每次提交前一定要多次检查！！！

每个项目都有一个main分支 属于公有分支， branch属于私有分支！！！

对于共有分支 只可往前走 不可往后走！！！

```
git clone #克隆到自己电脑
git remote -v #查看
git remote add upstream your_url #添加上游仓库

#加功能
git checkout -b 分支名 #切换进分支

git add . #. 代表所有文件
git commit -m "add(member):your name" #规范一下 “”内容无所谓
git push -u origin 分支名 #第一次push

#如果别人在你提交代码之前 就commit了 这会导致版本不一致 需要先更新下本地版本

git fetch upstream #从上游更新一下最新代码

git merge upstream/main #合并远程的最新版本代码到自己的分支中

git push
```

8）版本回退
```
git reset <your_file> #从暂缓区移除

git restore --staged <your_file> #更安全的操作 只会将其从暂缓区移除 不再commit这个文件

git checkout HEAD <your_file> #HEAD 表示最近一次commmit（需要注意 这个操作会丢失硬盘上所有的修改 需谨慎）

git reset --soft HEAD~1 #撤销local git中的commit，~1代表之前一个 2 3 

git reset HEAD ~1 等价于 git reset --mixed HEAD~1 #同时撤销 add和commit 只保留硬盘上的更改

git reset --hard HEAD~1 #全删 回到初始
假设说 误操作了hard 想找回
	1. git reflog
		 d9f8f32 (HEAD -> main) HEAD@{0}: reset: moving to HEAD~1
		 6b7a1c8 HEAD@{1}: commit: Add important feature
	找到误提交那个head 这里就是HEAD@{1} hash 6b7a1c8
	2. git reset --hard 6b7a1c8


针对共有分支 更推荐另一种形式的撤销

	git revert HEAD 
	
	#不会删除历史记录 生成一个新的commit 这个commit是对上一个commit的逆操作
	优势：可以撤销任意一次的提交 而且可以操作撤销共有分支的操作 很推荐

	git push

针对个人分支(除自己以外没有任何人用)
	
	git reset --hard HEAD~1 #砍掉一整个commit
	
	git push -f #如果想同步到远端 必须使用-f 千万别在共有分支整嗷
```
9）branch
```
git clone your_url
git checkout -b branch_name
git branch #check branch
git branch -a #check branch in remote repo
git add -u/ .
git commit -m "branch_create" #下面不可以直接push 因为没有上游
git push -u origin HEAD
```
10）断开远程仓库链接
```
git remote remove origin
```
11）***由于a,b都在更新 导致push有冲突
```
git switch main
git pull
git switch new-feature
git rebase main #会询问需要哪个版本
git rebase --continue #解决一个冲突 如果还有冲突需要再次git rebase rebase完之后会接在main后面
git push -f #由于造成顺序错乱 所以需要-f 十分不建议在主分支上做
```
12）删除branch 只会删除自己本地上的 github的不会
```
git branch -d <branch_name>
```
13）merge 1 结合 12 在github上直接操作

14）merge 2 先在自己的电脑上merge之后再push到github
```
git pull #保证a,b进度一样 这里慎用git push -f 会版本回退 更新内容消失
#如果发生冲突参考 12
git switch main
git merge <branch_name>
git commit #在vim查看下merge的信息 ：wq保存退出
git push
```
祝你工作顺利嗷 别被穿小鞋:)
