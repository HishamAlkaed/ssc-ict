# UbiOps Server Usage Dashboard (Streamlit)

A modern, interactive dashboard for monitoring UbiOps deployment metrics and performance using Streamlit.

## 🚀 Features

- **Real-time Metrics**: Monitor deployment requests, performance, and resource usage
- **Interactive Visualizations**: Beautiful charts using Plotly
- **Flexible Time Ranges**: Select custom date and time ranges
- **Multiple Aggregation Periods**: From 1 minute to 1 day aggregation
- **Project Selection**: Switch between different UbiOps projects
- **KPI Dashboard**: Key performance indicators at a glance
- **Responsive Design**: Works on desktop and mobile devices

## 📋 Prerequisites

- Python 3.8 or higher
- UbiOps API access and tokens
- Required Python packages (see requirements.txt)

## 🛠️ Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd streamlit_dashboard
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API tokens:**
   Edit the `app.py` file and update the `API_TOKENS` dictionary with your UbiOps API tokens:
   ```python
   API_TOKENS = {
       "poc": "Token YOUR_POC_TOKEN_HERE",
       "chat": "Token YOUR_CHAT_TOKEN_HERE"
   }
   ```

4. **Update deployment labels (if needed):**
   If you need to filter metrics for specific deployment versions, update the `GEMMA_DEPLOYMENT_LABEL_POC` variable.

## 🚀 Running the Application

1. **Start the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser:**
   The application will automatically open in your default browser at `http://localhost:8501`

## 📊 How to Use

### 1. **Select Project**
   - Choose between 'poc (Gemma)' or 'chat (alles)' projects

### 2. **Set Time Range**
   - Select start and end dates
   - Choose specific times for precise monitoring

### 3. **Configure Aggregation**
   - Choose aggregation period (1 minute to 1 day)
   - Smaller periods show more detail, larger periods show trends

### 4. **Select Metrics**
   - **Deployment Metrics**: Requests, duration, volume, queue times, etc.
   - **Custom Metrics**: Token counts, time to first token, etc.

### 5. **Update Dashboard**
   - Click "Update Dashboard" to fetch and display data
   - View KPIs and interactive charts

## 📈 Available Metrics

### Deployment Metrics
- **Requests**: Number of requests per minute
- **Failed Requests**: Number of failed requests per minute
- **Request Duration**: Average request completion time
- **Input/Output Volume**: Data transfer volumes in bytes
- **Express/Batch Queue Time**: Queue processing times
- **Credits**: Usage of credits
- **Network In/Out**: Inbound and outbound traffic

### Custom Metrics
- **Token Counts**: Prompt, completion, and total tokens
- **Cumulative Tokens**: Running totals of token usage
- **Time to First Token**: Response latency measurement

## 🎨 Features

- **KPI Cards**: Key performance indicators with visual indicators
- **Interactive Charts**: Hover for details, zoom, pan, and download
- **Responsive Layout**: Adapts to different screen sizes
- **Error Handling**: Graceful handling of API errors and missing data
- **Loading States**: Visual feedback during data fetching

## 🔧 Configuration

### API Configuration
- Update `API_TOKENS` with your UbiOps tokens
- Modify `configuration.host` if using a different UbiOps instance
- Adjust deployment labels for specific filtering

### Styling
- Custom CSS for modern appearance
- Gradient headers and card-based layout
- Consistent color scheme and typography

## 🐛 Troubleshooting

### Common Issues

1. **API Connection Errors**
   - Verify your API tokens are correct
   - Check network connectivity
   - Ensure UbiOps service is accessible

2. **No Data Displayed**
   - Check if the selected time range has data
   - Verify metric names are correct
   - Ensure deployment labels match your setup

3. **Performance Issues**
   - Reduce time range for faster loading
   - Use larger aggregation periods
   - Limit the number of selected metrics

### Error Messages
- **"API Error fetching..."**: Check API tokens and connectivity
- **"No data available..."**: Verify time range and metric selection
- **"An error occurred..."**: Check console for detailed error information

## 📝 Customization

### Adding New Metrics
1. Add metric name to the appropriate dictionary in the sidebar
2. Update the `DEPLOYMENT_METRICS` dictionary with unit and description
3. Ensure the metric is available in your UbiOps project

### Modifying Charts
- Edit the `display_metrics_charts` function
- Customize chart types, colors, and layouts
- Add new visualization types as needed

### Styling Changes
- Modify the CSS in the `st.markdown` section
- Update colors, fonts, and layout
- Add custom components as needed

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review UbiOps API documentation
3. Create an issue in the repository

---

**Note**: Keep your API tokens secure and never commit them to version control. Consider using environment variables for production deployments. 