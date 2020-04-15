import React, { useEffect, useState } from 'react';
import './App.css';
import { LandingPage, SearchBar, AboutPage, ResultPage } from './components';

function App() {
  let [pop, setPop] = useState(false);

  useEffect(()=>{
    let body = document.getElementsByTagName('body')[0];
    setTimeout(()=>{
      body.classList.toggle('fade');
    }, 500)
  });

  return (
    <div className="App" id="section1">
      <LandingPage>
        <SearchBar pop={pop} setPop={setPop} />
      </LandingPage>
      {pop && <ResultPage />}
      <AboutPage />
    </div>
  );
}

export default App;
