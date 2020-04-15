import React, { useEffect } from 'react';
import layer3 from '../img/layer3.png';
import layer1 from '../img/layer1.png';
import layer2 from '../img/layer2.png';
import '../styles/LandingPage.css';

const LandingPage = React.memo(({children}) =>{
    function adjustTextLayer(){
        let w = document.documentElement.clientWidth;
        let offset = 450;
        let textLayer = document.getElementById("text-layer1");
        textLayer.style.left = ((w - offset)/2) + 'px';

        let navBody = document.querySelector(".nav-body");

        if(w <= 870 && window.pageYOffset >= 350){
            navBody.classList.add("hide");
        }else{
            navBody.classList.remove("hide");
        }
    }
    useEffect(() => {
        let layer1 = document.getElementById("layer1");
        let layer0 = document.getElementById("layer0");
        let scroll = window.pageYOffset;
        let textLayer = document.getElementById("text-layer1");
        let logo = document.querySelector(".nav-logo a");
        let about = document.querySelector(".nav-rightItem a");
        adjustTextLayer();

        document.addEventListener('scroll', (e)=>{
            let offset = window.pageYOffset;
            scroll = offset;
            layer1.style.top = scroll/3 +'px';
            layer0.style.top = scroll/2 + 'px';
            textLayer.style.top = scroll / 1.5 + 'px';
            if(scroll >= 600){
                logo.style.color = 'white';
                about.style.color = 'white';
            }else{
                logo.style.color = 'black';
                about.style.color = 'black';
            }

            let navBody = document.querySelector(".nav-body");

            if(document.documentElement.clientWidth <= 870 && scroll >= 350){
                navBody.classList.add("hide");
            }else{
                navBody.classList.remove("hide");
            }
        });

        window.addEventListener("resize", adjustTextLayer);
      });
    return (
        <div className="parallax-section">
            <div className="nav-body stay-top">
                <div className="nav-logo">
                    <a href="#section1">Skillset Analyser</a>
                </div>
                <div className="nav-rightItem">
                    <a href="#section2">About This Page</a>
                </div>
            </div>
            <div id="text-layer1">
                <p id="p1">Skillset Analysis</p>
                <p id="p2">Find out what skill sets  <br></br> you need for your future career!</p>
            </div>
            {children}
            <img id="layer0" src={layer3} alt="parallax-0"/>
            <img id="layer1" src={layer1} alt="parallax-1"/>
            <img id="layer2" src={layer2} alt="parallax-2"/>
        </div>  
    );
});

export default LandingPage;