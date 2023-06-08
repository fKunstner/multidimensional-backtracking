

mkdir -p zip_staging
cp -r code/ zip_staging/

cd zip_staging/code
rm -r .idea
rm -r private
rm .flake8
rm .gitignore
rm .isort.cfg
rm -r scripts/__pycache__
rm scripts/results/*
rm -r src/precsearch/__pycache__
rm -r src/precsearch/*/__pycache__
rm -r src/precsearch/*/*/__pycache__
rm -r src/precsearch/*/*/*/__pycache__

cd ..

[ -e code_supplementary.zip ] && rm code_supplementary.zip
zip -r code_supplementary.zip code

cd ..

mv zip_staging/code_supplementary.zip .

rm -r zip_staging