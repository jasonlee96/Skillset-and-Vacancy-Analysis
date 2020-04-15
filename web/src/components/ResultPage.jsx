import React, { useEffect } from 'react';
import "../styles/ResultPage.css";

const ResultPage = () => {
    useEffect(() => {
        let box = document.getElementById("section3");
        let position = box.getBoundingClientRect().top + 50;

        window.scrollTo({
            top: window.pageYOffset + position,
            behavior: "smooth"
        });
    })

    return(
        <div className="result-container" id="section3">
            <div className="result-box shadow"></div>
        </div>
    );
}

export default ResultPage;