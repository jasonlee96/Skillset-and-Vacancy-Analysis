import React, { useEffect, useState } from 'react';
import "../styles/SearchBar.css";

const SearchBar = ({ pop, setPop}) => {
    function search(){
        if(pop){
            setPop(false);
            setTimeout(()=>{
                setPop(true);
            }, 100);
        }else{
            setPop(true);
        }
    }

    function adjustTextLayer(){
        let w = document.documentElement.clientWidth;
        let offset = 600;
        let searchBar = document.querySelector(".bar");
        searchBar.style.left = ((w - offset)/2) + 'px';
    }

    function activeDropdown(){
        let dropdown = document.getElementsByClassName("dropdown-content")[0];
        dropdown.classList.toggle("show");
    }

    function selectDropdown(i){
        let dropdown = document.getElementsByClassName("dropdown")[0];
        dropdown.value = i;
    }

    useEffect(()=>{
        let searchBar = document.querySelector(".bar");
        let scroll = window.pageYOffset;
        adjustTextLayer();

        window.addEventListener("resize", adjustTextLayer);

        document.addEventListener('scroll', (e)=>{
            let offset = window.pageYOffset;
            scroll = offset;
            searchBar.style.top = 460 + scroll / 1.5 + 'px';

            if(offset >= 500){
                searchBar.classList.add("hide");
            }else{
                searchBar.classList.remove("hide");
            }
        });
    })

    return(
        <div className="bar shadow">
            <div className="dropdown-box" onClick={activeDropdown}>
                <select name="type" className="dropdown">
                    <option value="0">Job Title</option>
                    <option value="1">Location</option>
                </select>
                <div className="dropdown-content shadow">
                    <div className="dropdown-item" onClick={()=> selectDropdown(0)}>
                        Job Title
                    </div>
                    <div className="dropdown-item" onClick={()=> selectDropdown(1)}>
                        Location
                    </div>
                </div>
            </div>
            <input className="text-field" placeholder="Keywords...." type="text"/>
            <button className="submit-btn" type="submit" onClick={search}>Search</button>
        </div>
    )
}

export default SearchBar;