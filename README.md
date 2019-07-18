# cmpt353duoduo



git clone https://github.com/ZiyiAn/DiningPal.git  rename

cd rename

heroku create diningpal-yourname 

git push heroku master

(git remote show heroku)

git remote set-url --add --push heroku https://github.com/ZiyiAn/DiningPal.git

git remote set-url --add --push heroku https://git.heroku.com/diningpal-yourname.git

git remote set-url heroku https://github.com/ZiyiAn/DiningPal.git

To CREATE and check into your branch:
git checkout -b yourbranchname

(git branch yourbranchname
git checkout yourbranchname)

(edit sth ie. git add .  git commit ...)

To push to YOUR OWN branch: 
git push heroku yourbranchname
(git push --set-upstream origin yourbranchname)

git fetch heroku gitbranchname 
(or with auto-merge:
git pull heroku gitbranchname)

To delete local branch: 
git branch -d branch_name
git branch -D branch_name (force to delete)
To delete remote branch: 
git push heroku --delete <branch_name>

To HARD RESET local branch:
(git commit -a -m "Saving my work, just in case"
git branch my-saved-work)

git fetch <repo_name>
git reset --hard <repo_name> / <branch_name>

