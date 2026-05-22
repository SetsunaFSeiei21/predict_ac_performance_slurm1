git init
git add .
git commit -F scripts/git/commit_message.txt
git branch -M main

if ! git remote | grep -qx 'origin'; then
    echo "Adding new remote origin: git@github.com:SetsunaFSeiei21/ACCFormer.git"
    git remote add origin git@github.com:SetsunaFSeiei21/ACCFormer.git
else
    echo "Remote 'origin' already exists - skipping addition"
    git remote set-url origin git@github.com:SetsunaFSeiei21/ACCFormer.git
fi

git push -u origin main