#list of usernames and corresponding working directories.
working_dir = {"nikoog":"/content/drive/My Drive/ENGSCI/Year 3/SUMMER 2020/APS360/",
               }

import os

class git_commands:
  def __init__(self, email, uname, init=True, repo_link="https://github.com/CarbonicKevin/APS360-2020Summer-Project.git"):
    self.email = email
    self.uname = uname
    self.repo  = repo_link
    self.repo_name = repo_link.split('/')[-1][:-4]

    try:
      self.dir = working_dir[uname]
    except:
      self.dir = input("path to the directory you'll be working in:")

    if os.path.isdir(self.dir):
      %cd $self.dir
    else:
      print("ERR: invalid directory specified")

    try:
      !git config --global user.email $email --quiet
      !git config --global user.name $uname --quiet

      if init:
        !git clone $self.repo

      %cd $self.repo_name
      !git pull

    except:
      print("ERR: failed to initialize github. check input values and retry.")

    break

  def pull(self):
    dir = !pwd
    if dir[0] != self.dir:
      %cd self.dir
    !git pull

  def commit(self, commit_msg, add_files):
    if add_files == 'all':
      !git add -A
    else:
      print("ERR: invalid input for add_files.")
      return -1

    if len(commit_msg) > 50:
      print("ERR: commit message too long.")
      return -1

    passwd = getpass()

    !git commit -m commit_msg
    !git remote add origin "https://{uname}:{passwd}github@github.com/{repo}.git"

    !git push -u origin master

    return 1
