import org.deidentifier.arx.ARXAnonymizer;
import org.deidentifier.arx.ARXConfiguration;
import org.deidentifier.arx.ARXResult;
import org.deidentifier.arx.AttributeType;
import org.deidentifier.arx.Data;
import org.deidentifier.arx.DataType;
import org.deidentifier.arx.AttributeType.Hierarchy;
import org.deidentifier.arx.AttributeType.MicroAggregationFunction;
// import org.deidentifier.arx.AttributeType.Hierarchy.DefaultHierarchy;
// import org.deidentifier.arx.aggregates.HierarchyBuilder;
// import org.deidentifier.arx.aggregates.HierarchyBuilderIntervalBased;
// import org.deidentifier.arx.aggregates.HierarchyBuilderRedactionBased;
// import org.deidentifier.arx.aggregates.HierarchyBuilderIntervalBased.Range;
// import org.deidentifier.arx.aggregates.HierarchyBuilderRedactionBased.Order;
import org.deidentifier.arx.criteria.KAnonymity;

// import java.io.ObjectInputFilter.Config;
import java.nio.charset.StandardCharsets;
import java.io.File;
import java.util.Scanner; 

public class micro_agg {
    public static void main(String[] args) throws Exception{

        // String input_file = args[1];
        
        int k = Integer.parseInt(args[0]);
		String input_file = args[1];
		double suppression = Double.parseDouble(args[2]);
        String type = args[3];

        Data data = Data.create(input_file, StandardCharsets.UTF_8, ',');

		File myobject = new File("hierarchy.txt");
        Scanner myReader = new Scanner(myobject);

        while (myReader.hasNextLine()) {
            String line = myReader.nextLine();
            String[] types = line.split(",");
            
			if (types[1].equals("Insensitive")) {
				data.getDefinition().setAttributeType(types[0], AttributeType.INSENSITIVE_ATTRIBUTE);
			} else if (types[1].equals("Identifying")) {
				data.getDefinition().setAttributeType(types[0], AttributeType.IDENTIFYING_ATTRIBUTE);
			} else if (types[1].equals("Quasi_identifying")){
				data.getDefinition().setAttributeType(types[0], AttributeType.QUASI_IDENTIFYING_ATTRIBUTE);
			} else if (types[1].equals("Sensitive")) {
				data.getDefinition().setAttributeType(types[0], AttributeType.SENSITIVE_ATTRIBUTE);
			} else {
				data.getDefinition().setAttributeType(types[0], Hierarchy.create(types[1], StandardCharsets.UTF_8, ','));
			} 
            
            if (types[2].equals("NaN")) {
                continue;
            } else if (types[2].equals("Integer")) {
                data.getDefinition().setDataType(types[0], DataType.INTEGER);
            } else if (types[2].equals("Decimal")) {
                data.getDefinition().setDataType(types[0], DataType.DECIMAL);
            } else if (types[2].equals("Ordered_String")) {
                data.getDefinition().setDataType(types[0], DataType.ORDERED_STRING);
            } else if (types[2].equals("String")) {
                data.getDefinition().setDataType(types[0], DataType.STRING);
            } else if (types[2].equals("Date")) {
                data.getDefinition().setDataType(types[0], DataType.DATE);
            } 

            if (types[3].equals("NaN")) {
                continue;
            } else if (types[3].equals("arithmic_mean")) {
                data.getDefinition().setMicroAggregationFunction(types[0], MicroAggregationFunction.createArithmeticMean());
            } else if (types[3].equals("geometric_mean")) {
                data.getDefinition().setMicroAggregationFunction(types[0], MicroAggregationFunction.createGeometricMean());
            } else if (types[3].equals("interval")) {
                data.getDefinition().setMicroAggregationFunction(types[0], MicroAggregationFunction.createInterval());
            } else if (types[3].equals("median")) {
                data.getDefinition().setMicroAggregationFunction(types[0], MicroAggregationFunction.createMedian());
            } else if (types[3].equals("mode")) {
                data.getDefinition().setMicroAggregationFunction(types[0], MicroAggregationFunction.createMode());
            } else if (types[3].equals("set")) {
                data.getDefinition().setMicroAggregationFunction(types[0], MicroAggregationFunction.createSet());
            }

        }
        myReader.close();

        ARXAnonymizer anonymizer = new ARXAnonymizer();
        ARXConfiguration config = ARXConfiguration.create();

        config.addPrivacyModel(new KAnonymity(k));
        config.setSuppressionLimit(suppression);
        config.setSuppressionAlwaysEnabled(false);

        ARXResult result = anonymizer.anonymize(data, config);

        File f = new File("Results_anonymisation/Micro_Anony_" + k + "_"+ type + ".csv");
		
		if(!f.exists() && !f.isDirectory()) { 
			result.getOutput(false).save("Results_anonymisation/Micro_Anony_" + k + "_"+ type + ".csv", ',');
		}
    
        // result.getOutput(false).save("Results_anonymisation/Micro_Anony_" + k + "_"+ type + ".csv", ',');
    }

}
