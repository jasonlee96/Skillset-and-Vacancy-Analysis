import React from 'react';
import "../styles/AboutPage.css";
import layer3 from '../img/about-layer.png';

const AboutPage = React.memo(() => {
    return(
        <div className="about-container" id="section2">
            <div className="about-background">
                <img src={layer3} alt="background"/>
            </div>
            <div className="about-box">
                <div className="title">
                    <span>About This Page</span>
                </div>
                <div className="about-content">
                    <p>It is an analysis system to reveal top trending skill set for any job position and vacancy demand at any location via using data mining method!</p>
                    <p>Author's GitHub: <span id="link">https://github.com/jasonlee96</span></p>
                </div>
            </div>
        </div>
    ); 
});

export default AboutPage;