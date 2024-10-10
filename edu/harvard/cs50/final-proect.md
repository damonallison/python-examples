# Final Project (API)

* install [asdf](https://asdf-vm.com/guide/introduction.html) - manages tool
  (i.e., `nodejs`) versions

* learn typescript (typing)
* learn eslint (lint)
* learn prettier (formatting)

* use [bruno](https://www.usebruno.com)
    * simple / git friendly (bru syntax)

```shell

# install nest
npm i -g @nestjs/cli

# create the api project (first time only)
nest new api

cd api

npm run test

# launch API on port 3000 (watching files)
npm run start:dev

# ensure api is up and running
curl -v http://localhost:3000

######################################
# sqlite
######################################

npm install @nestjs/typeorm typeorm sqlite3


```