#!/bin/bash

# got to project
function gotoproj() {
  cd $PROJECT_HOME/$1
}

# got to sanskrit project
function gotosans() {
  cd $PROJECT_HOME/sanskrit/$1
}

function stjupyter() {
  cd $PROJECT_HOME/$1
  conda activate ${2:-sans}
  jupyter notebook
}