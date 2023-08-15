// 这个假设的函数用于从Flask获取数据，返回一个像{'Math': 85, 'History': 90}这样的对象
// function fetchSubjectsData() {
//     // 这里是获取数据的代码
// }

// 颜色获取函数
function getChartColorsArray(e) {
    if (null !== document.getElementById(e)) {
        e = document.getElementById(e).getAttribute("data-colors");
        return JSON.parse(e).map(function (e) {
            var t = e.replace(" ", "");
            return -1 === t.indexOf(",") ? getComputedStyle(document.documentElement).getPropertyValue(t) || t : 2 == (e = e.split(",")).length ? "rgba(" + getComputedStyle(document.documentElement).getPropertyValue(e[0]) + "," + e[1] + ")" : t;
        });
    }
}

// 假设的subjectsData，你应该使用实际的方法获取这个数据
var subjectsData = {
    // 'Math': 85,
    // 'History': 90,
    // // ... 其他学科
};

// ApexCharts配置
var options = {
    series: [{
        name: "Scores",
        data: Object.values(subjectsData)
    }],
    chart: {
        width: 1500,  // 设定宽度
        type: "area",
        height: 450,
        zoom: {
            enabled: false
        }
    },
    dataLabels: {
        enabled: false
    },
    stroke: {
        curve: "smooth" // 设置为曲线
    },
    title: {
        text: "Subjects Scores",
        align: "left",
        style: {
            fontWeight: 500
        }
    },
    subtitle: {
        text: "Percentage Scores",
        align: "left"
    },
    labels: Object.keys(subjectsData),
    xaxis: {
        type: "category",
        categories: Object.keys(subjectsData) // 学科名称
    },
    yaxis: {
        min: 0,
        max: 100,
        labels: {
            formatter: function (val) {
                return val + "%"; // 添加百分比符号
            }
        }
    },
    legend: {
        horizontalAlign: "left"
    },
    colors: getChartColorsArray("area_chart_basic")
};

var chart = new ApexCharts(document.querySelector("#area_chart_basic"), options);
chart.render();
