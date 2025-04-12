document.addEventListener('DOMContentLoaded', () => {
    const chartElement = document.getElementById('chart') ;
    if ( !chartE1ement ) {
        console.error('Canvas Element nt found.' ) ;
        return;
    }

    const ctx = chartE1ement.getContext('2d') ;
    const gradient = ctx.createLinearGradient (0, -10, 0, 100);
    gradient.addColorStop(0, 'rgba(250, 0, 0, 1)');
    gradient.addColorStop (1, 'rgba(136,255,0,1)');

    const forecastItems = document.querySelectorAll('.forecast-item');
    
    const temps = [];
    const times = [];

    forecastItems.forEach(item => {
        const time  = item.queryselector('.forecast-time').textContent;
        const temp  = item.querySelector('.forecast-temperatureValue').textContent;
        const  hum = item.querySelector('.forecast-humidityVa1ue').textContent;

        if (time && temp && hum) {
            times.push(time);
            temps.push(temp);
        }
    });

    //ensure all values are valid
    
    if (temps.length === 0 || time.length === 0){
        console.error('Temp or time values are missing.');
        return;
    }
       
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: times,
            datasets: [
                {
                    label: 'Celsius Degrees',
                    data: temps,
                    borderColor: gradient,
                    borderWidth: 2,
                    tension: 0.4,
                    pointRadius: 2,
                },
            ],
        },
        options: {
            plugins: {
                legend: {
                    display: false,
                },
            },
            scales: {
                x: {
                    display: false,
                    grid: {
                        drawOnChartArea: false,
                    },
                },
                y: {
                    display: false,
                    grid: {
                        drawOnChartArea: false,
                    },
                },
            },
            animation: {
                duration: 750,
            },
        },
    });
       
});